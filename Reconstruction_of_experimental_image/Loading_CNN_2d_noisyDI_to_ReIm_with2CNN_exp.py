import os
import time
import math
from sympy import symbols, hermite, expand
from scipy.special import hermite as scipy_hermite
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.special import hermite as scipy_hermite
from scipy.signal import fftconvolve
from PIL import Image
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * sigma ** 2))

def photo_current(m, n, sigma, X, Y, image_matrix):
    import math
    from scipy.special import hermite as scipy_hermite

    dx = (X[0, -1] - X[0, 0]) / X.shape[1]
    dy = (Y[-1, 0] - Y[0, 0]) / Y.shape[0]
    Hm = scipy_hermite(m)(X / ( sigma))
    Hn = scipy_hermite(n)(Y / ( sigma))
    Bases = 1 / (2 ** (m + n) * math.factorial(m) * math.factorial(n))**0.5 * Hm * Hn * Sai(X, Y, sigma)
    return np.sum(Bases * image_matrix) * dx * dy

def photo_current_object(m, n, sigma, X, Y, object_matrix):
    import math
    dx = (X[0, -1] - X[0, 0]) / X.shape[1]
    dy = (Y[-1, 0] - Y[0, 0]) / Y.shape[0]
    Bases = 1 / (math.sqrt(math.factorial(m) * math.factorial(n))) * (X / (2 * sigma)) ** m * (Y / (2 * sigma)) ** n * np.exp(
        -1 * (X ** 2 + Y ** 2) / (8 * sigma ** 2))
    return np.sum(Bases * object_matrix) * dx * dy

# Define the variable
u = symbols('u')

def alpha(m, k):
    H = hermite(m, u)
    H_expanded = expand(H)
    coeffs = H_expanded.as_coefficients_dict()
    return coeffs.get(u**k, 0)

def process_single_sample(object_matrix_100, all_exp_DI, sigma,
                          m_max, n_max, X, Y, nx, ny,
                          Hx, Hy, exp_factor, C, denom):

    im = object_matrix_100
    if im.max() == im.min():
        im = np.zeros_like(im);
        im[50, 50] = 1  # fallback Dirac
    else:
        im = (im - im.min()) / (im.max() - im.min())
    obj61 = im
    obj61 = (obj61 - obj61.min()) / (obj61.max() - obj61.min())
    obj61 = (obj61 > 0.5).astype(float)

    # 0B) build the noisy direct image
    # centered PSF
    cx, cy = nx // 2, ny // 2
    Sai_grid = Sai(X - X[cx, cy], Y - Y[cx, cy], sigma)
    print("obj61 shape:", obj61.shape)
    print("Sai_grid shape:", Sai_grid.shape)

    DI = fftconvolve(obj61, Sai_grid, mode='same')
    DI = (DI - DI.min()) / (DI.max() - DI.min())

    # A) image photocurrents P_image[k,l]
    P_image = np.zeros((m_max, n_max), float)
    for k in range(m_max):
        for l in range(n_max):
            P_image[k, l] = photo_current(k, l, sigma, X, Y, DI)

    # B) contract to get all beta_image[m,n]
    beta_image = np.tensordot(C, P_image, axes=([2, 3], [0, 1]))  # (m_max,n_max)

    # C) normalize them
    norm_image = beta_image * denom  # (m_max,n_max)

    # D) reconstruct ReIm_target
    ReIm_target = np.zeros((nx, ny), complex)
    for m in range(m_max):
        for n in range(n_max):
            basis = (Hy[n][:, None] * Hx[m][None, :]) * exp_factor
            ReIm_target += norm_image[m, n] * basis

    ReIm_target = ReIm_target / ReIm_target.max()
    ReIm_target[ReIm_target < 0] = 0
    ReIm_target = np.real(ReIm_target)

    return all_exp_DI, ReIm_target

class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DenoisingCNN0(nn.Module):
    def __init__(self):
        super(DenoisingCNN0, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def process_single_sample_wrapper(args):
    return process_single_sample(*args)

def main():
    # Start timing
    tic = time.time()
    from skimage.transform import resize

    # Load DenoisingCNN
    denoising1_model = DenoisingCNN0()
    denoising1_model.load_state_dict(torch.load(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\NN\Fine_tunning\5.28k_CNN_2d_exp_to_2d_ReIm\conv1d_simple_nn.pth'))
    denoising1_model.eval()

    # Load DenoisingCNN
    denoising2_model = DenoisingCNN()
    denoising2_model.load_state_dict(torch.load(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\NN\Fine_tunning\5.28k_CNN_2d_exp_to_2d_ReIm_withsecond_CNN\conv1d_simple_nn.pth'))
    denoising2_model.eval()

    # Parameters
    sigma = 0.15

    m_max = 20
    n_max = 20

    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    number_of_mesh_x = 100
    number_of_mesh_y = 100

    x = np.linspace(x_min, x_max, number_of_mesh_x)
    y = np.linspace(y_min, y_max, number_of_mesh_y)
    X, Y = np.meshgrid(x, y)

    from scipy.special import eval_hermite

    # 1A) Hermite–Gaussian bases
    xi = x / (2 ** 0.5 * sigma)
    yj = y / (2 ** 0.5 * sigma)
    Hx = np.stack([eval_hermite(m, xi) for m in range(m_max)], axis=0).astype(np.float64)
    Hy = np.stack([eval_hermite(n, yj) for n in range(n_max)], axis=0).astype(np.float64)
    exp_factor = np.exp(-(X ** 2 + Y ** 2) / (4 * sigma ** 2)).astype(np.float64)

    # 1B) Mixing tensor C[m,n,k,l] = √(k! l!) α(m,k) α(n,l)
    fact = np.array([math.factorial(i) for i in range(max(m_max, n_max))], dtype=float)
    alpha_tab = np.zeros((m_max, m_max), float)
    for m in range(m_max):
        for k in range(m_max):
            alpha_tab[m, k] = float(alpha(m, k))
    C = np.zeros((m_max, n_max, m_max, n_max), float)
    for m in range(m_max):
        for n in range(n_max):
            for k in range(m_max):
                for l in range(n_max):
                    C[m, n, k, l] = math.sqrt(fact[k] * fact[l]) * alpha_tab[m, k] * alpha_tab[n, l]

    # 1C) Denominator denom[m,n] = 1/(π σ² · 2^(m+n+2) m! n!)
    prefac = 1 / (np.pi * sigma ** 2)
    denom = np.zeros((m_max, n_max), float)
    for m in range(m_max):
        for n in range(n_max):
            denom[m, n] = prefac / (2 ** (m + n + 2) * fact[m] * fact[n])

    image_size = (100, 100)

    num_objects = 1

    folder_path = r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Results'
    all_objects = np.zeros((num_objects, image_size[0], image_size[1]), dtype=np.float32)
    all_exp_DI = np.zeros((num_objects, image_size[0], image_size[1]), dtype=np.float32)

    obj_cnt = 0
    exp_cnt = 0
    single_object = []

    fn_obj = f"Obj_0_52.xlsx"
    full_path = os.path.join(folder_path, fn_obj)  # Combine folder and file name
    df = pd.read_excel(full_path, header=None, index_col=None)
    O = df.values
    O = (O - O.min())/(O.max()-O.min())
    mat_obj = resize(O, (100, 100), anti_aliasing=True)
    mat_obj = (mat_obj - mat_obj.min())/(mat_obj.max()-mat_obj.min())
    all_objects[0]=mat_obj
    single_obj_60 = resize(mat_obj, (60, 60), anti_aliasing=True)
    single_obj_60 = (single_obj_60 - single_obj_60.min())/(single_obj_60.max()-single_obj_60.min())
    single_obj_60 = (single_obj_60 > 0.5).astype(float)


    fn_exp = f"Exp_0_52.xlsx"
    full_path = os.path.join(folder_path, fn_exp)  # Combine folder and file name
    df = pd.read_excel(full_path, header=None, index_col=None)
    D = df.values
    D = (D - D.min())/(D.max()-D.min())
    mat_exp = resize(D, (100, 100), anti_aliasing=True)
    mat_exp = (mat_exp - mat_exp.min())/(mat_exp.max()-mat_exp.min())
    all_exp_DI[0]=mat_exp
    single_DI_60 = resize(mat_exp, (60, 60), anti_aliasing=True)
    single_DI_60 = (single_DI_60 - single_DI_60.min())/(single_DI_60.max()-single_DI_60.min())
    DI_tensor_60 = torch.from_numpy(single_DI_60).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        DI_denoised_60 = denoising1_model(DI_tensor_60).squeeze().numpy()
    DI_denoised_tensor_60 = torch.from_numpy(DI_denoised_60).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        DI_pre_60 = denoising2_model(DI_denoised_tensor_60).squeeze().numpy()

    print("Starting parallel processing...")

    args = [
        (all_objects[i], all_exp_DI[i], sigma, m_max, n_max, X, Y,
         number_of_mesh_x, number_of_mesh_y,
         Hx, Hy, exp_factor, C, denom)
        for i in range(num_objects)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_sample_wrapper, args))

    exp_DI_list, ReIm_list = zip(*results)
    # Resize to 60x60
    exp_DI_resized = [resize(im, (60, 60), anti_aliasing=True) for im in exp_DI_list]
    ReIm_resized = [resize(im, (60, 60), anti_aliasing=True) for im in ReIm_list]



    # Ensure everything is float
    object_matrix = resize(all_objects[0], (60, 60), anti_aliasing=True)
    E_DI = all_exp_DI[0]
    E_recovered = DI_pre_60
    object_matrix=(object_matrix - object_matrix.min()) / (object_matrix.max() - object_matrix.min())
    object_matrix = (object_matrix > 0.5).astype(float)
    GT = object_matrix.astype(float)
    DI = resize(E_DI, (60, 60), anti_aliasing=True).astype(float)
    MS = E_recovered.astype(float)

    # Helper to compute each metric
    def compute_metrics(name, img, reference):
        dr = reference.max() - reference.min()
        return {
            'Method': name,
            'PSNR(dB):H-B': psnr(reference, img, data_range=dr),
            'SSIM:H-B': ssim(reference, img, data_range=dr),
            'RSE(normRMSE):L-B': np.linalg.norm(img - reference) / np.linalg.norm(reference),
            'RSP(Pearsonr):H-B': np.corrcoef(reference.ravel(), img.ravel())[0, 1],
            'StdDev(norm):H-B': np.std(img) / np.std(reference)
        }

    # Compute for both methods
    results = [
        compute_metrics('Direct Imaging', DI, GT),
        compute_metrics('Mode Sorting', MS, GT)
    ]

    df = pd.DataFrame(results).set_index('Method')

    # Print with full values
    print(df.round(6).to_string())

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(14, 3))

    # Inside your plotting loop
    ax0.contourf(resize(all_objects[0], (60, 60), anti_aliasing=True), levels=50)
    ax0.set_title("Object")
    ax1.contourf(abs(all_exp_DI[0]), levels=50)
    ax1.set_title("Input: noisy image")
    ax2.contourf(abs(ReIm_resized[0]), levels=50)
    ax2.set_title("Ground Truth: noiseless image")
    ax3.contourf(abs(DI_pre_60), levels=50)
    ax3.set_title("NN Prediction")
    plt.show()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()