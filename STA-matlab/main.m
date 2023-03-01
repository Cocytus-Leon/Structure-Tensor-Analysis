clear;clc
%% Set paths
imFile = 'E:\Project\Matlab\NisslST-main\0.1DAPI_CNN_331_351_max.jpg';
maskFile = 'E:\Project\Matlab\NisslST-main\mask.jpg';
nissl_st_dir = 'E:\Project\Matlab\NisslST-main';
addpath(genpath(nissl_st_dir));

%% Set analysis flags and options
ds_factor = 0.2;% Downsample factor for downsampling the image
rho = 15; % measured in pixels (~15 microns)
sigma = 3; % Gaussian kernel for smoothing the image before calculating Nissl-ST. This parameter is typically 0, but for this Allen Brain Atlas dataset, it seems that sigma=3 yields better results.
add_peak_orientations = false; % mark peak orientation or not
nPeaks = 1; % Number of peak orientations to extract. In fact, we typically just take the first peak.
sample_near_cells = true; % Only extract orientations as close as 'rho' to cells (to avoid measuring empty spaces)
para.Step = 15; % intensity of orientation
para.scl = 5; % length of orientation
para.width = 1; % width of orientation
degree = 1; % angle accuracy

%% STA process
im = imread(imFile);
im = im2gray(im);
im_mask = mask_read(maskFile);

im = imresize(im,ds_factor);
im_mask = imresize(im_mask,ds_factor);

[theta_mean, theta_vec, ~] = img_sta_figures(im,im_mask,rho,sigma,sample_near_cells,para);
[theta_peaks,pks_height] = find_peak_orientations(theta_vec,nPeaks);

%% Draw results
figure('color','w')
theta_plot = theta_vec(:);
theta_plot = [theta_plot; theta_plot(theta_plot<0)+180; theta_plot(theta_plot>0)-180];
theta_plot = theta_plot.*pi/180;
bins_edges = (0:degree:360);
bin_vals = bins_edges;
bin_vals = [bin_vals, 0, 360];
bin_vals(bin_vals>180) = bin_vals(bin_vals>180)-180;
bin_rgbTmp = vals2colormap(bin_vals, hsv, []);
bin_rgbTmp(1,:) = [];
bin_rgbTmp(end-1:end,:) = [];
bins_edges = bins_edges./180*pi;

for i = 1:360/degree
    bins_edge = bins_edges(i:i+1);
    h = polarhistogram(theta_plot,'BinEdges',bins_edge,'Normalization','count','linew',degree);
    hold all
    h.DisplayStyle = 'bar'; % Or 'stairs' for unfilled histogram
    h.EdgeColor = bin_rgbTmp(i,:);
    h.FaceColor = bin_rgbTmp(i,:);
    h.FaceAlpha = 1;
end

set(gca,'RTick',[])
set(gca,'ThetaTickLabel',[]) % don't display degree values
set(gca,'linew',2)
set(gca,'ThetaColor','w')
rl = rlim;
set(gca,'color','k')

%% Add the peak orientations
if add_peak_orientations
    mu_deg = theta_peaks;
    for mI = 1
        if mI == 1;frmt = '-r';else;frmt = '--r'; end
        hplt = polarplot([mu_deg(mI),mu_deg(mI)-180]*pi/180,[1,1].*rl(2),frmt,'linew',degree);
        % Get orientation RGB color
        vals = mu_deg(mI);
        vals = [vals, 0, 180];
        rgbTmp = vals2colormap(vals, hsv, []);
        rgbTmp(end-1:end,:) = [];
        hplt.Color = rgbTmp;
    end
end