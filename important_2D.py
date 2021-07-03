# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
img = plt.imread("bbb.jpg")
#R = img[:, :, 0]
#G = img[:, :, 1]
#B = img[:, :, 2]
#img = ((0.2989 * R + 0.5870 * G + 0.1140 * B)*255).astype(np.float16)
# %%
print(img.shape)
print(img.dtype)
print(img.min())
print(img.max())
# %% 将unit16图像像素值转到0-255区间段


def norm_img(img):
    max_v = img.max()
    min_v = img.min()
    img = (img - min_v) / (max_v - min_v)
    img = (img*255).astype(np.uint8)
    #img = img.astype(np.int16)
    #img[img < 0] = 0
    return img


# %% 将原图像素值整改到0-255区间
img = 255-norm_img(img)
# %%
plt.imshow(img, cmap=plt.cm.gray)
# %%
print(img.shape)
print(img.dtype)
print(img.min())
print(img.max())
print(np.median(img))
# %%
# * 创建核函数


def CreateGaussianKernel(sigma, normalizeflag):
    R = np.ceil(2*sigma*np.sqrt(np.log(10)))
    [X, Y] = np.meshgrid(np.arange(-R, R+1, 1), np.arange(-R, R+1, 1))
    h = (1/(2*np.pi*sigma*sigma))*np.exp(-(X*X + Y*Y)/(2*sigma*sigma))
    if normalizeflag == 1:
        h = h/np.sum(h)
    return h


def CreateDoGxDoGyKernel(sigma):
    R = np.ceil(3.57160625*sigma)
    [X, Y] = np.meshgrid(np.arange(-R, R+1, 1), np.arange(-R, R+1, 1))
    DoGx = -(X/(2*np.pi*sigma**4))*np.exp(-(X*X + Y*Y)/(2*sigma**2))
    DoGy = -(Y/(2*np.pi*sigma**4))*np.exp(-(X*X + Y*Y)/(2*sigma**2))
    return DoGx, DoGy


# %%
# * Standard deviation of derivative-of-gaussian (DoG) kernels [pixel]
sigma_DoG = 0.5
# * Standard deviation of Gaussian kernel [pixel]
sigma_Gauss = 2
GaussianKernel = CreateGaussianKernel(sigma_Gauss, 1)
DoGxKernel, DoGyKernel = CreateDoGxDoGyKernel(sigma_DoG)
# %%
[R, C] = np.shape(img)
Tensor_Orientation = np.zeros([R, C])
Tensor_AI = np.zeros([R, C])
# %%
dImage_dx = correlate2d(img, DoGxKernel, 'same')
dImage_dy = correlate2d(img, DoGyKernel, 'same')
# %%
Ixx = dImage_dx*dImage_dx
Ixy = dImage_dx*dImage_dy
Iyy = dImage_dy*dImage_dy
# %%
Jxx = correlate2d(Ixx, GaussianKernel, 'same')
Jxy = correlate2d(Ixy, GaussianKernel, 'same')
Jyy = correlate2d(Iyy, GaussianKernel, 'same')
# %%
dummyEigenvalues11 = np.zeros([R, C])
dummyEigenvalues22 = np.zeros([R, C])
# %%
for rr in range(R):
    for cc in range(C):
        J_rr_cc = np.array([[Jxx[rr, cc], Jxy[rr, cc]],
                           [Jxy[rr, cc], Jyy[rr, cc]]])
        Vals, featurevector = np.linalg.eig(J_rr_cc)
        dummyEigenvalues11[rr, cc] = Vals[0]
        dummyEigenvalues22[rr, cc] = Vals[1]
# %%
# Apply Bigun and Grandlund formula (ref [2]) providing anles in [-pi;pi]
bufferPhi = 0.5*np.angle((Jyy - Jxx) + 1j*2*Jxy)
# %%
# Remap negative angles in the positive range, so that angles will be in [0; pi]
bufferPhi[bufferPhi < 0] = np.angle(
    np.exp(1j*bufferPhi[bufferPhi < 0])*np.exp(1j*np.pi))
# %%
# Save the orientation map in radians
Tensor_Orientation = bufferPhi
# %%
# Save the anisotropy map
Tensor_AI = abs(dummyEigenvalues11 - dummyEigenvalues22) / \
    abs(dummyEigenvalues11 + dummyEigenvalues22)
# %%
HueThetaZero = 0
HueThetaPi = 1
H = HueThetaZero + (1/np.pi)*(HueThetaPi - HueThetaZero)*Tensor_Orientation
S = Tensor_AI
V = 1 - img/255
# %%
image_HSV = np.dstack((H, S, V))
print(image_HSV.shape)
print(image_HSV.dtype)
# %%


def HSV2RGB(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h = 6*h
    k = np.floor(h)
    p = h-k
    t = 1-s
    n = 1-s*p
    p = 1-(s*(1-p))
    kc = (k == 0)
    r = kc
    g = kc*p
    b = kc*t
    kc = (k == 1)
    r = r + kc*n
    g = g + kc
    b = b + kc*t
    kc = (k == 2)
    r = r + kc*t
    g = g + kc
    b = b + kc*p
    kc = (k == 3)
    r = r + kc*t
    g = g + kc*n
    b = b + kc
    kc = (k == 4)
    r = r + kc*p
    g = g + kc*t
    b = b + kc
    kc = (k == 5)
    r = r + kc
    g = g + kc*t
    b = b + kc*n
    kc = (k == 6)
    r = r + kc
    g = g + kc*p
    b = b + kc*t
    out = np.dstack((r, g, b))
    out[:, :, 0] = v/np.max(out)*out[:, :, 0]
    out[:, :, 1] = v/np.max(out)*out[:, :, 1]
    out[:, :, 2] = v/np.max(out)*out[:, :, 2]
    return out


# %%
image_RGB = HSV2RGB(image_HSV)
# %%
image_OUT = (255*image_RGB).astype(np.uint8)
# %%
plt.imshow(image_OUT)
# %%
plt.imsave('OUT_2.jpg', image_OUT)
