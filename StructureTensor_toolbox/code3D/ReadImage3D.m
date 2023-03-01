function [FinalImage,Siz] = ReadImage3D(ImName)
% % This code is to read 3D image as a matrix
% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %        wenxing84@gmail.com.

InfoImage= imfinfo(ImName);
mImage   = InfoImage(1).Width;
nImage   = InfoImage(1).Height;
NumberImages=length(InfoImage);
FinalImage=zeros(nImage,mImage,NumberImages,'uint16');
for i=1:NumberImages
   FinalImage(:,:,i)=imread(ImName,'Index',i);
end
Siz = [nImage,mImage,NumberImages];

