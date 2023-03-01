function Grad_ker=RotVarGradient(Dim,K_siz)
% % This code computes the gradient which has a good rotation invariance property.
% % It is based on the work.
% % J. Weickert and H. Scharr, "A scheme for coherence-enhancing diffusion
% % filtering with optimized rotation invariance", 
% % Journal of Visual Communication and Image Representation,
% % 13(1):103-118, 2002.
% % Input Dim -- Values{2, 3}. The dimension of image, 2D or 3D
% %       K_siz- the size of kernel. Values{2,3,4,5} (see also above reference)
% % Example: 
% % Dim=3; K_siz=3; Grad_ker=RotVarGradient(Dim,K_siz);
% %  Then, Grad_ker is kernel for n1 x n2 x n3 image.

% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %        wenxing84@gmail.com.

if Dim~=2&&Dim~=3; error('Codes is for 2D or 3D cases!'); end
if K_siz<2||K_siz>5; error('Codes is for kernel size {2,3,4,5}!'); end
switch K_siz
    case 2  %%%%2x2 kernel 
        Dx=[1 -1]; Qx=[1 1]/2;
    case 3  %%%%3x3 kernel
        if Dim==2; p1=0.182962; else p1=0.174654; end
        Dx=[1 0 -1]/2; Qx =[p1 1-2*p1 p1];
    case 4   %%%%4x4 kernel
        if Dim==2; p1=-0.087584; else p1=-0.075116; end
        Dx=[-1 27 -27 1]/24; Qx=[p1 0.5-p1 0.5-p1 p1]; 
    case 5   %%%%5x5 kernel
        if Dim==2; 
            p1=-0.043881;  p2=0.164561; 
        else
            p1=-0.0384;  p2=0.150654;
        end
        Dx=[-1 8 0 -8 1]/12;  Qx=[p1 p2 1-2*(p1+p2) p2 p1]; 
end
Dy=Dx'; Qy=Qx'; Dz=reshape(Dx,[1 1 K_siz]); Qz=reshape(Qx,[1 1 K_siz]);
if Dim==2
    Grad_ker.Gx=kron(Dx,Qy); 
    Grad_ker.Gy=kron(Dy,Qx);
else
    Grad_ker.Gx=superkron(Dx,Qy,Qz); 
    Grad_ker.Gy=superkron(Dy,Qz,Qx);
    Grad_ker.Gz=superkron(Dz,Qx,Qy);
end

