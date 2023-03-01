function mask = mask_read(mask_path)
mask = imread(mask_path);
if size(mask,3)>1
    mask = rgb2gray(mask);
end
mask = 255-mask;
mask = logical(mask);
end