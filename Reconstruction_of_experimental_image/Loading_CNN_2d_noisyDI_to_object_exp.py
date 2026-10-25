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

def process_single_sample_wrapper(args):
    return process_single_sample(*args)

def main():
    # Start timing
    tic = time.time()
    from skimage.transform import resize

    # Load DenoisingCNN
    denoising_model = DenoisingCNN()
    denoising_model.load_state_dict(torch.load(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\NN\Fine_tunning\5.28k_CNN_2d_exp_to_2d_object\conv1d_simple_nn.pth'))
    denoising_model.eval()

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
    O = (O - O.min()) / (O.max() - O.min())
    mat_obj = resize(O, (100, 100), anti_aliasing=True)
    mat_obj = (mat_obj - mat_obj.min()) / (mat_obj.max() - mat_obj.min())
    all_objects[0] = mat_obj
    single_obj_60 = resize(mat_obj, (60, 60), anti_aliasing=True)
    single_obj_60 = (single_obj_60 - single_obj_60.min()) / (single_obj_60.max() - single_obj_60.min())
    single_obj_60 = (single_obj_60 > 0.5).astype(float)

    fn_exp = f"Exp_0_52.xlsx"
    full_path = os.path.join(folder_path, fn_exp)  # Combine folder and file name
    df = pd.read_excel(full_path, header=None, index_col=None)
    D = df.values
    D = (D - D.min()) / (D.max() - D.min())
    mat_exp = resize(D, (100, 100), anti_aliasing=True)
    mat_exp = (mat_exp - mat_exp.min()) / (mat_exp.max() - mat_exp.min())
    all_exp_DI[0] = mat_exp
    single_DI_60 = resize(mat_exp, (60, 60), anti_aliasing=True)
    single_DI_60 = (single_DI_60 - single_DI_60.min()) / (single_DI_60.max() - single_DI_60.min())
    DI_tensor_60 = torch.from_numpy(single_DI_60).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        obj_pre_60 = denoising_model(DI_tensor_60).squeeze().numpy()


    # Ensure everything is float
    object_matrix = resize(all_objects[0], (60, 60), anti_aliasing=True)
    E_DI = all_exp_DI[0]
    E_recovered = obj_pre_60
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
        compute_metrics('CNN', MS, GT)
    ]

    df = pd.DataFrame(results).set_index('Method')

    # Print with full values
    print(df.round(6).to_string())

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))

    # Inside your plotting loop
    ax0.contourf(resize(all_objects[0], (60, 60), anti_aliasing=True), levels=50)
    ax0.set_title("Object")
    ax1.contourf(abs(all_exp_DI[0]), levels=50)
    ax1.set_title("Input: noisy image")
    ax2.contourf(abs(obj_pre_60), levels=50)
    ax2.set_title("NN Prediction")
    plt.show()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()