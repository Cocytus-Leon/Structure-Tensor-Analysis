function ConvInfo=ConvexHullCells3D(im,Thresh,dlt)
% % This code compute the convexhull of 3D image.
% % Input: im    -- nx*ny*nz original 3D image
% %        Thresh-- thresholding the original image as binary image
% %        dlt   -- the ratio of x,y,z direction (useful for 3D biological image)
% % Output: ConvInfo.imconv  --- convexhull of 3D image
% %         ConvInfo.imBound --- the boundary of 3D convexhull
% %         ConvInfo.Vt      --- pixels composing binary image
% %         ConvInfo.indx    --- x index of convexhull boundary
% %         ConvInfo.indy    --- y index of convexhull boundary
% %         ConvInfo.indz    --- z index of convexhull boundary
% %          
% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %  wenxing84@gmail.com.
time = cputime; 
im  = im/max(im(:)); 
[nx,ny,nz] = size(im);  
gk  = gaussian_kernel3D(nx,ny,nz,1,dlt); 
imb_thresh = ifftn(fftn(gk).*fftn(im)) > Thresh;       
clear im  gk


%%%%%%%%%%%%%%%%%%%%%%%% Vertex and convex hull
Ind = find(imb_thresh>1-eps);
indz= ceil(Ind/(nx*ny));
indy= ceil(Ind/nx-(indz-1)*ny);
indx= Ind-(indz-1)*nx*ny-(indy-1)*nx;
Ind = [indx,indy,indz];
Vt  = convhulln(Ind);

%%%%%%%%%%%%%%%%%%%%%%%% boundary and convexhull
[X,Y,Z]= meshgrid(1:ny,1:nx,1:nz);
imconv = ones(nx,ny,nz)>0;
numvet = size(Vt);
pa = Ind(Vt(:,1),:); %%%% point a
pb = Ind(Vt(:,2),:); %%%% point b
pc = Ind(Vt(:,3),:); %%%% point c and normal vector NVA,NVB,NVC
for i=1:numvet(1)  %%% for each tri-patch plane
    Nv = cross(pa(i,:)-pb(i,:),pb(i,:)-pc(i,:));
    Nv = Nv/norm(Nv);
    z  = Nv(1)*Y+Nv(2)*X+Nv(3)*Z-Nv*pa(i,:)';  %%% Plane
    imconv(z<0)=0;
end

%  figure(1);
%  VisualData3D(permute(imconv,[2,1,3]));  axis([0 nx 0 ny 0 nz]); alpha(0.5)
%  close all
clear X Y Z z Ind

imBound=zeros(size(imconv));  %%%% boundary of convhull
for i=1:nz
    imBound(:,:,i) = edge(imconv(:,:,i),'prewitt');  
end

ConvInfo.imconv     = imconv;      clear imconv
ConvInfo.imBound    = imBound;     clear imBound
ConvInfo.imb_thresh = imb_thresh;  clear imb_thresh
ConvInfo.Vt         = Vt;          clear Vt
ConvInfo.indx       = indx;        clear indx
ConvInfo.indy       = indy;        clear indy
ConvInfo.indz       = indz;        clear indz 


%%%%%%%%%%%%%%%%%%%%%%%% distance to to bound in convex hull
D    = bwdist(ConvInfo.imBound);
grad = gradient3D(D); 
Tep  = sqrt(grad(:,:,:,1).^2+grad(:,:,:,2).^2+grad(:,:,:,3).^2)+eps;
grad(:,:,:,1)=grad(:,:,:,1)./Tep;  %%%%%%% normalized gradients
grad(:,:,:,2)=grad(:,:,:,2)./Tep;
grad(:,:,:,3)=grad(:,:,:,3)./Tep;

clear Tep

DistInfo.D   = ConvInfo.imconv.*D;
DistInfo.GDx = grad(:,:,:,1);
DistInfo.GDy = grad(:,:,:,2);
DistInfo.GDz = grad(:,:,:,3);

fprintf('CPU time for convexhull: %4.2f s\n',cputime-time);




