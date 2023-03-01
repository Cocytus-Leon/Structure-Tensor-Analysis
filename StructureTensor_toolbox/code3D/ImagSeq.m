function ImagSeq(U)
% % This code display an n1 x n2 x n3 matrix as image (frame by frame)
% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.
% %        wenxing84@gmail.com.

siz=size(U);
nz=siz(3);
close all

for i=1:nz
    h = figure(1); imshow(U(:,:,i),'border','tigh');
    saveas(h,['im3D',int2str(i),'.bmp']);
end