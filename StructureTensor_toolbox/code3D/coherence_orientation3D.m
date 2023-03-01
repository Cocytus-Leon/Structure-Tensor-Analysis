function EigInfo=coherence_orientation3D(im,rho,dlt)     
%   This function generates the eigenvalues and the eigenvectors of the
%   3D structure tensor matrix.
%   INPUT:
%   im: test 3D image
%   rho: parameter for the filter K_\rho
%   dlt: ratios of x,y,z directions of original image dlt=[1 1 1] for
%   synthetic data

%   OUTPUT: 
%   EigInfo.J_rho --- the structure tensor        
%   EigInfo.nu1,EigInfo.nu2,EigInfo.nu3 --- eigenvalues of J_rho(x)
%   EigInfo.w1, EigInfo.w2, EigInfo.w3  --- eigenvectors of J_rho(x)
%
% Developer : Wenxing Zhang, wxzh1984@126.com, July 2014.
time = cputime;
sigma= 0; 

im = im/max(im(:));
[nx,ny,nz] = size(im);
[J_rho,im_sigma] = structure_tensor3D(im,sigma,rho,dlt);  %% tensor matrix

clear im

w1 = zeros(nx,ny,nz,3); w2  = zeros(nx,ny,nz,3);  w3  = zeros(nx,ny,nz,3);
nu1= zeros(nx,ny,nz);   nu2 = zeros(nx,ny,nz);    nu3 = zeros(nx,ny,nz);
tic
for k=1:nz
    for i=1:nx
        for j=1:ny
            M=[J_rho(i,j,k,1),J_rho(i,j,k,2),J_rho(i,j,k,3); ...
               J_rho(i,j,k,4),J_rho(i,j,k,5),J_rho(i,j,k,6); ...
               J_rho(i,j,k,7),J_rho(i,j,k,8),J_rho(i,j,k,9)];
            [V,D]=eig(M);
            nu1(i,j,k)=D(1,1); nu2(i,j,k)=D(2,2); nu3(i,j,k)=D(3,3);
            w1(i,j,k,:)=V(:,1);w2(i,j,k,:)=V(:,2);w3(i,j,k,:)=V(:,3);
        end
    end
end
toc



%%%%%%%%%%%%%%%%%%%%%%%%%Algorithm2 for Eigvalue %%%%%%%%%%%%%%%%%%%%%%%%
% MaxV=max(abs(J_rho(:)));
% J_rho=J_rho./MaxV;
% tic
% c0=J_rho(:,:,:,1).*J_rho(:,:,:,5).*J_rho(:,:,:,9) ...
%     +2*J_rho(:,:,:,2).*J_rho(:,:,:,3).*J_rho(:,:,:,6) ...
%     -J_rho(:,:,:,1).*J_rho(:,:,:,6).^2 ...
%     -J_rho(:,:,:,5).*J_rho(:,:,:,3).^2 ...
%     -J_rho(:,:,:,9).*J_rho(:,:,:,2).^2;
% c1=J_rho(:,:,:,1).*J_rho(:,:,:,5)-J_rho(:,:,:,2).^2 ...
%     +J_rho(:,:,:,1).*J_rho(:,:,:,9)-J_rho(:,:,:,3).^2 ...
%     +J_rho(:,:,:,5).*J_rho(:,:,:,9)-J_rho(:,:,:,6).^2;
% c2=J_rho(:,:,:,1)+J_rho(:,:,:,5)+J_rho(:,:,:,9);
% 
% a=c1-c2.^2/3; b=c1.*c2/3-2*c2.^3/27-c0; q=b.^2/4+a.^3/27;
% rrhoo=sqrt(-a/3); theta=atan2(sqrt(-q),-b/2)/3;
% cs=cos(theta);
% sn=sin(theta);
% 
% nu11=c2/3+2*rrhoo.*cs;
% nu22=c2/3-rrhoo.*(cs+sqrt(3)*sn);
% nu33=c2/3-rrhoo.*(cs-sqrt(3)*sn);
% toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Normalization: (nu1<nu2<nu3)
Tep = sqrt(w1(:,:,:,1).^2+w1(:,:,:,2).^2+w1(:,:,:,3).^2);
w1  = w1./repmat(Tep,[1,1,1,3]);      

Tep = sqrt(w2(:,:,:,1).^2+w2(:,:,:,2).^2+w2(:,:,:,3).^2);
w2  = w2./repmat(Tep,[1,1,1,3]);    

Tep = sqrt(w3(:,:,:,1).^2+w3(:,:,:,2).^2+w3(:,:,:,3).^2);
w3  = w3./repmat(Tep,[1,1,1,3]);

   
    
EigInfo.nu1 = nu1; EigInfo.w1 = w1;    clear nu1  w1  Tep
EigInfo.nu2 = nu2; EigInfo.w2 = w2;    clear nu2  w2
EigInfo.nu3 = nu3; EigInfo.w3 = w3;    clear nu3  w3
EigInfo.im_sigma = im_sigma;           clear im_sigma
EigInfo.J_rho    = J_rho;              clear J_rho


fprintf('CPU time for tensors: %4.2f s\n',cputime-time);





end


