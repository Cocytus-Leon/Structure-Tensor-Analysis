function [Tensor,im_sigma]=structure_tensor3D(im,sigma,rho,dlt)
%   This function computes the structure tensor matrix associated to the 
%   3D smoothed image im.
%   INPUT:
%   im: m x n x slices matrix   
%   rho: the standard deviation of the gaussian kernel applied to the
%   strcture tensor to average the directions of the eigenvectors.
%   OUTPUT:
%   T_rho: the structure tensor a 3*n by 3*m by slices matrix. T_rho is a 
%   9 blocks 2D matrix [A1,A2,A3;A4,A5,A6;A7,A8,A9] each block of size 
%   [n,m,slices]. For example,the first block should contain the first 
%   component of the structure tensor for each pixel of the image.
% In order to find parallel structures we need to replace the gradient by
% its tensor product so that the structure descriptor is invariant under
% sign changes:

% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %            wenxing84@gmail.com.

% Gaussian kernel convolution applied to the image 
[nx,ny,nz]=size(im); 
if sigma==0
    im_sigma = im;   
else
    gk_sigma = gaussian_kernel3D(nx,ny,nz,sigma,dlt);    
    im_sigma = ifftn(fftn(gk_sigma).*fftn(im));   
end
clear gk_sigma  im

grad = zeros(nx,ny,nz,3);        
Grad_ker=RotVarGradient(3,3);    %%% gradient u, i.e., \nabla u:        
grad(:,:,:,2) = -imfilter(im_sigma,Grad_ker.Gx); 
grad(:,:,:,1) = -imfilter(im_sigma,Grad_ker.Gy); 
grad(:,:,:,3) = -imfilter(im_sigma,Grad_ker.Gz); 

Tensor = zeros(nx,ny,nz,9);
for i=1:3
    for j=i:3
        Tensor(:,:,:,(i-1)*3+j)= grad(:,:,:,i).*grad(:,:,:,j);
    end
end     

for i=2:3
    for j=1:i-1
        Tensor(:,:,:,(i-1)*3+j)=Tensor(:,:,:,(j-1)*3+i);
    end
end

clear grad

% Componentwise convolution with another gaussian kernel
Fgk_rho= fftn(gaussian_kernel3D(nx,ny,nz,rho,dlt));  
for i = 1:9
    Tensor(:,:,:,i)=ifftn(Fgk_rho.*fftn(Tensor(:,:,:,i))); 
end

