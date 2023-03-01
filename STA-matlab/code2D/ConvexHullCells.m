function ConvInfo=ConvexHullCells(im,Thresh)
%   This function computes the convexhull of a 2D binary image
%   INPUT:
%   im: 2D original image. 
%   Thresh: thresholding the original image as binary image
%
%   OUTPUT:
%   ConvInfo.imconv: the binary image representing image content
%   ConvInfo.Vt:     the pixels composing the convexhull
%   ConvInfo.indx:   x index of convexhull boundary
%   ConvInfo.indy:   y index of convexhull boundary.

% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.
% %        wenxing84@gmail.com.


[nx,ny]=size(im);
im  = im/max(im(:)); 
gk  = gaussian_kernel(ny,nx,3);
imb = ifft2(fft2(gk).*fft2(im));
imb_thresh = imb>Thresh;

clear gk im  imb

%%%%%%%%%%%%%%%%%%%%%%%% Vertex and convex hull
[indx,indy]= find(imb_thresh>1-eps);
Vt   = convhull(indx,indy,'simplify',true);
grdx = indx(Vt(1:end-1))-indx(Vt(2:end));
grdy = indy(Vt(1:end-1))-indy(Vt(2:end));
ngrdx= grdx./sqrt(grdx.^2+grdy.^2);  %%% normalize
ngrdy= grdy./sqrt(grdx.^2+grdy.^2);    


%%%%%%%%%%%%%%%%%%%%%%%% boundary and convexhull
[X,Y]  = meshgrid(1:ny,1:nx);
imconv = ones(nx,ny);
imBound= zeros(nx,ny);   
D    = (nx*ny)^2*ones(nx,ny);  %%% Distance function.
GDx  = zeros(nx,ny);           %%% normal to the tangent at boundary
GDy  = zeros(nx,ny);

for i=1:length(Vt)-1
    z = ngrdy(i)*(Y-indx(Vt(i)))-ngrdx(i)*(X-indy(Vt(i)));    
    
    %%% convexhull
    imconv(z<0)=0;   
    
    %%% convexhull boundary
    xa=max(indy(Vt(i)),indy(Vt(i+1))); xb=min(indy(Vt(i)),indy(Vt(i+1))); 
    ya=max(indx(Vt(i)),indx(Vt(i+1))); yb=min(indx(Vt(i)),indx(Vt(i+1)));      
    Tep = zeros(nx,ny); 
    Tep( X>=xb & X<=xa & Y>=yb & Y<=ya&abs(z)<=0.5)=1; 
    Tep(abs(z)<=1.)=1;    %%%%active line for polyhedron
     
    imBound(Tep>0)=1;  
    GDx(Tep>0)=-ngrdx(i);   %%% Normal vector of boundary point
    GDy(Tep>0)=ngrdy(i);
    
    %%% Distance function
    Tep = D;   D = min(Tep,abs(z));
    GDx(Tep>D)=-ngrdx(i); %%% Normal vector of point in convexhull
    GDy(Tep>D)= ngrdy(i); 
end
   

%%%%%%%%% Another method for distance function %%%%%%%%%%%%%%%%%
clear X Y z  Tep            
ConvInfo.imconv     = imconv;        clear imconv 
ConvInfo.Vt         = Vt;            clear Vt 
ConvInfo.indx       = indx;          clear indx
ConvInfo.indy       = indy;          clear indy  