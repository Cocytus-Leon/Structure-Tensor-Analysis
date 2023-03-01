function [theta_mean, theta_vec, thresh] = img_sta_figures(im,mask_main,rho,sigma,sample_near_cells,para)

if size(im,3)>1
    im = rgb2gray(im);
end
im_orig = im;
im = 255-im;

%%
figure('color','k')
im_mask = im_orig.*uint8(mask_main);
imshow(im_mask)

%% Get threshold for binarizing sub-image
% This is later used to construct the brain mask
bins_centers = (0:254)+0.5;
counts = hist(im(:),bins_centers);
try
[thresh,~] = otsuthresh(counts);
catch % If MATLAB version is <2017b, otsuthresh will fail
thresh = graythresh(im);
end

%% Create sampling mask
if sample_near_cells
    mask = im2bw(im,thresh);
    mask = imfill(mask,'holes');
    %mask = 1-mask;
    mask = imdilate(mask,strel('disk',rho));
else
    mask = ones(size(im));
end
mask = logical(mask);
mask = mask&mask_main;
%% Compute structral tensor 
EigInfo = coherence_orientation_with_sigma(double(im),rho,sigma); 
    ConvInfo.imconv = ones(size(im));
    plot_near_cells = true;
    DisplayImage_withMaskFlag(im_orig ,EigInfo,ConvInfo,para,mask,plot_near_cells);
    set(gca,'color','k')

%% Calculate coherence of the 1st eigenvector, as a measure of anisotropy over the entire sub-image
eigvec_weighted = EigInfo.w2;
eigvec_weighted = apply_mask(eigvec_weighted,mask);

if any(~isreal(eigvec_weighted))
    theta_mean = nan;
    theta_vec = nan;
    return
end

% Get theta_vec (all thetas, one per pixel in the mask)
[theta_vec,~] = cart2pol(eigvec_weighted(:,1),eigvec_weighted(:,2)); % result is in [-pi pi]
theta_vec = theta_vec*180/pi; % Now everything is in [-pi,pi]
theta_vec(theta_vec<0) = theta_vec(theta_vec<0)+180; % Now everything will be in [0,180], which is good for the colormap

%% Calculate the mean eigenvector, to get the angle
% eigvec_mean = mean(eigvec_weighted);
% Or try like this:
eigvec_mean = sum(eigvec_weighted);
eigvec_mean = eigvec_mean./norm(eigvec_mean);
[theta_mean,~] = cart2pol(eigvec_mean(1),eigvec_mean(2)); % Returns [-pi,pi]
theta_mean = theta_mean*180/pi;
if theta_mean<0
    theta_mean = theta_mean+180; % theta will be in [0,180])
end
theta_mean = round(theta_mean); % A score over the 180 colormap values

end

%% Additional functions
function img_out = apply_mask(img,mask)
    % This will only take the values of img inside mask. Output is a
    % nx1 vector. If img has more dimensions, output is nxm.
    img_out = [];
    for dI = 1:size(img,3)
        tmp = img(:,:,dI);
        tmp = tmp(mask);
        img_out(:,dI) = tmp(:); % (nxm) X d matrix. The original img dimensions are not reduced to long vectors
    end    
end