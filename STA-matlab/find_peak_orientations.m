function [theta_peaks,pks_height] = find_peak_orientations(theta_vec,nPeaks)
% Input is in [0 180], so we expand it below. 
    % Convert to radians
    theta_vec = theta_vec/180*pi;
    % Apply ksdensity to smooth the histogram and find the optimal
    % smoothing kernel bandwidth
    [~,~,bw] = ksdensity(theta_vec,0:pi/180:pi); % bw is the smoothing kernel bandwidth
    % Initialize
    theta_peaks = nan(1,nPeaks);
    pks_height = nan(1,nPeaks);
    % Use ksdensity to smooth histogram and find peaks
    theta_tmp = [theta_vec-pi; theta_vec; theta_vec+pi]; % Repeat the data, so we don't interpret peaks near 0 as two peaks
    xi = -pi : pi/180 : 2*pi;
    [f] = ksdensity(theta_tmp,xi,'BandWidth',bw);
        
    % Normalize to maximal peak of 1
    f = f./max(f);
    % Find peaks
    [pks,locs] = findpeaks(f);
    if isempty(pks)
        return
    end
    % Remove peaks outside of [0,pi]
    idx = xi(locs)>0 & xi(locs)<pi;
    locs = locs(idx);
    pks = f(locs);
    % Sort peaks in descending order (consider ordering according to width
    % multiplied by peak, to estimate the integral under the peak). See
    % findpeaks for the width.
    [pks,I] = sort(pks,'descend');
    locs = locs(I);
    for pI = 1:length(locs)
        theta_peaks(pI) = xi(locs(pI));
        pks_height(pI) = pks(pI);
    end
    pks_height = pks_height (1:nPeaks);
    theta_peaks = theta_peaks(1:nPeaks);
    theta_peaks = theta_peaks*180/pi;
end