import math
import numpy as np
from sympy import symbols, hermite, expand
from scipy.special import hermite as scipy_hermite
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image
import time
# Define the variable
u = symbols('u')
import scipy.ndimage
img = Image.open(r'K:\Dropbox\Mohammad_Lab_Computer\Sub-diffraction imaging\Paper\Incoherent\Biomedical\3_400.png').convert('L')
object_matrix = np.array(img, dtype=float)

object_matrix = (object_matrix-np.min(object_matrix))/(np.max(object_matrix)-np.min(object_matrix))
object_matrix = object_matrix + 1e-2
#object_matrix[object_matrix < 0.4] = 1e-17

def Sai(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x**2 + y**2) / (2 * sigma ** 2))

def final_DI_image(object_matrix):
    # Parameters
    photon_number = 10 ** 17
    sigma = 0.75 / 1000

    x_min = -0.00454545454545*8
    x_max = 0.00454545454545*8
    y_min = -0.00454545454545*8
    y_max = 0.00454545454545*8
    number_of_mesh_x = 8*50
    number_of_mesh_y = 8*50
    # Create meshgrid for x and y
    x = np.linspace(x_min, x_max, number_of_mesh_x)
    y = np.linspace(y_min, y_max, number_of_mesh_y)
    X, Y = np.meshgrid(x, y)
    E_grid = object_matrix
    Sai_grid = Sai(X - X[number_of_mesh_x // 2, number_of_mesh_x // 2],
                   Y - Y[number_of_mesh_y // 2, number_of_mesh_y // 2], sigma)
    E_DI = fftconvolve(E_grid, Sai_grid, mode='same')
    return E_DI

DI_final_array = []
DI_final_array = final_DI_image(object_matrix)

x_min = -0.00454545454545*8
x_max = 0.00454545454545*8
y_min = -0.00454545454545*8
y_max = 0.00454545454545*8
number_of_mesh_x = 400
number_of_mesh_y = 400

# Create meshgrid for x and y
x = np.linspace(x_min, x_max, number_of_mesh_x)
y = np.linspace(y_min, y_max, number_of_mesh_y)
X, Y = np.meshgrid(x, y)

contour_di = plt.contourf(x*1000, y*1000, DI_final_array, levels=200, cmap='viridis')
plt.colorbar(contour_di)
plt.title('Direct Imaging')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()

contour_obj = plt.contourf(x*1000, y*1000, object_matrix, levels=200, cmap='viridis')
plt.colorbar(contour_obj)
plt.title('Object')
plt.xlabel('x')
plt.ylabel('y')
plt.show()