% function DisplayImage3D(im,EigInfo,ConvInfo,para)
%
% This function displays the different informations returned by function
% coherence_orientation. 
%
% INPUT : 
% - im : 2D image
% - EigInfo : 
% - ConvInfo : 
% - para : 
%    para.Step : stepsize for oreintation field display
%    para.scl : length of orientation segments
%
% OUTPUT : 
% - Figures will pop up 1) ground truth image, 2) convexhull, 3)
% orientation field, 4) anisotropoy map.
%
% Developer:  Wenxing Zhang, wenxing84@gmail.com, July 2014
function DisplayImage3D(im,EigInfo,ConvInfo,para)

%%%%%%%%% plots parameters
if isfield(para,'Step');    Step=para.Step;    else Step=10;   end

%%%%%%%%%%% Image Info and convex hull Info.
imconv   = ConvInfo.imconv;
Vt       = ConvInfo.Vt;
indx     = ConvInfo.indx;    
indy     = ConvInfo.indy;
indz     = ConvInfo.indz;            clear  ConvInfo
[nx,ny,nz]= size(im);

%%%%%%%%%%%%%% Distance function and Tensor info.
nu1  = EigInfo.nu1; w1 = EigInfo.w1;   
nu2  = EigInfo.nu2; w2 = EigInfo.w2;
nu3  = EigInfo.nu3; w3 = EigInfo.w3;      
J_rho= EigInfo.J_rho;
clear EigInfo DistInfo

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% displays %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; %%%%% 3D Image 
subplot(121); VisualData3D(im);    axis equal
title(['truth:',num2str(nx),'*',num2str(ny),'*',num2str(nz)]);

subplot(122); %%%%%%% convexhull
trisurf(Vt,indy,indx,indz,'Edgecolor','none','facecolor','g'); 
alpha(0.5); camlight; lighting gouraud; view(3); 
hold off; axis equal;  title('convHull'); grid off

s1=1:Step:nx; s2=1:Step:ny; s3=1:Step:nz; 
[cx,cy,cz] = meshgrid(s1,s2,s3);
[X,Y,Z]    = meshgrid(1:ny,1:nx,1:nz);
figure;  %%%% orientations
subplot(221); VisualData3D(im);  hold on
h=coneplot(X,Y,Z,w1(:,:,:,2).*imconv,w1(:,:,:,1).*imconv,w1(:,:,:,3).*imconv,...
    cy,cx,cz,5);  axis equal; set(h,'FaceColor','green','EdgeColor','none')
view(3); camlight; lighting phong;
title('tesnsor vector (w1)'); hold off

subplot(222); VisualData3D(im);  hold on
h=coneplot(X,Y,Z,w2(:,:,:,2).*imconv,w2(:,:,:,1).*imconv,w2(:,:,:,3).*imconv,...
    cy,cx,cz,5);axis equal; set(h,'FaceColor','green','EdgeColor','none')
view(3); camlight; lighting phong;
title('tesnsor vector (w2)'); hold off

subplot(223); VisualData3D(im);  hold on
h=coneplot(X,Y,Z,w3(:,:,:,2).*imconv,w3(:,:,:,1).*imconv,w3(:,:,:,3).*imconv,...
    cy,cx,cz,5); axis  equal; set(h,'FaceColor','green','EdgeColor','none')
view(3); camlight; lighting phong;
title('tesnsor vector (w3)'); hold off


figure;   %%%%% Anisotropy
subplot(121); Tep=sqrt(nu3./nu1).*imconv; imshow(Tep(:,:,fix(nz/2)),[]); 
colormap jet; colorbar; title('\sigma_3/\sigma_1(center slice)')
subplot(122); Tep=sqrt(nu2./nu1).*imconv; imshow(Tep(:,:,fix(nz/2)),[]); 
colormap jet; colorbar; title('\sigma_2/\sigma_1(center slice)')
    










     









