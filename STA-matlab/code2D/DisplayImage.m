% function DisplayImage(im,EigInfo,ConvInfo,para)
%
% This function displays the different informations returned by function
% coherence_orientation. 
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

function DisplayImage(im,EigInfo,ConvInfo,para)
 
%%%%%%%%% plots parameters
if isfield(para,'Step');    Step   =para.Step;    else Step   =20;   end
if isfield(para,'scl');     scl    =para.scl;     else scl  =10;    end
               
%%%%%%%%%%% Image Info and convex hull Info.
[nx,ny]    = size(im);
im         = im/max(im(:)); %%%%% 
imconv     = ConvInfo.imconv;  

%%%%%%%%%%%%%% Distance function and Tensor info.
nu1 = EigInfo.nu1; w1 = EigInfo.w1;   
nu2 = EigInfo.nu2; w2 = EigInfo.w2;
J_rho = EigInfo.J_rho;

%%%%%%%%%%%%% cell center, direction and radii info.
A =zeros(nx,ny); A(1:Step:nx,1:Step:ny)=1;
[i,j] = find(imconv.*A>0);
Synt.Centre = [j,i]';
        
Centre= Synt.Centre; %%% Synthetic and real data with ground-truth center
if median(Centre(:))<1;  Centre=Centre*max(nx,ny);  end
Cent  = fix(Centre(1,:))*nx+fix(Centre(2,:))+1; 
Centrex = Centre(1,:);  Centrey = Centre(2,:); 

clear EigInfo  ConvInfo  DistInfo  Synt

%%%%%%%%%%%%%%%%%%%%%%%%%%% displays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; imshow(im,[]); colormap gray; caxis([0.1 .5]) ; title(['truth image',num2str(nx),'*',num2str(ny)]);
figure; imshow(imconv,[]); colormap gray; title('convHull');hold on        

%%%%%%%%%%% comparisons the resutls
figure;   %%% orientations
a1=w1(:,:,2); a2=w1(:,:,1);  
imshow(im,[]); colormap gray;caxis([0.1 .5]);  hold on;
   P1=kron([1;1],Centrex)+scl*kron([1;-1],a1(Cent));
   P2=kron([1;1],Centrey)+scl*kron([1;-1],a2(Cent));
   plot(P1,P2,'g','linewidth',1); title('tensor orientation');hold off

figure; %%%% anisotropy
aniso=sqrt(nu1./nu2); aniso(imconv<0.5)=1; 
imshow(aniso,[]); 
colorbar('location','eastoutside'); title('anisotropy')
colormap jet; caxis([0.58 1.02])   
    





