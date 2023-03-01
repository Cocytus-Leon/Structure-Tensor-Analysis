function gk=gaussian_kernel3D(nx,ny,nz,sigma,dlt)
%function gk=gaussian_kernel_3D(nx,ny,nz,sigma,dlt)
%   this function computes the 3D gaussian kernel with sigma as a standard 
%   deviation. the output is an nx by ny by nz matrix.  
%%%% dlt=[deltax,deltay,deltaz] is the pixels resolutions
% % 
% %  Written by Dr. Pierre Weiss, ITAV, Toulouse, France
% %  pierre.armand.weiss@gmail.com; 

xgv  = linspace((-nx/2+1)*dlt(1),nx/2*dlt(1),nx); 
ygv  = linspace((-ny/2+1)*dlt(2),ny/2*dlt(2),ny); 
zgv  = linspace((-nz/2+1)*dlt(3),nz/2*dlt(3),nz);
[X,Y,Z]= meshgrid(ygv,xgv,zgv);
gk   = exp(-(X.^2+Y.^2+Z.^2)/(2*sigma^2));
gk   = fftshift(gk/sum(gk(:)));

end