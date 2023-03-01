function grad=gradient3D(u)
%function grad=gradient3D(u)
%   this function computes the 3D gradient of a 3D image u. grad is a
%   size(u) by 3 matrix.
% % This code aims to display the orientations and anisotropy of structure
% % Tensor based algorithm

% %  Written by Dr. Pierre Weiss, ITAV, Toulouse, France, 2014.4. % 
% % Email: pierre.armand.weiss@gmail.com; 

    [nx,ny,nz]=size(u);
    grad=zeros(nx,ny,nz,3);
    grad(1:end-1,:,:,1)=u(2:end,:,:)-u(1:end-1,:,:);
    grad(:,1:end-1,:,2)=u(:,2:end,:)-u(:,1:end-1,:);
    grad(:,:,1:end-1,3)=u(:,:,2:end)-u(:,:,1:end-1);
end