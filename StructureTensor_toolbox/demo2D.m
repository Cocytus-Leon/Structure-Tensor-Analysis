clc
clear 
close all
addpath ./code2D  ./TestData
rand('seed',0); randn('state',0)
%% %%%%%%%%% Test data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ImagName = 'Synth2D';       %%%%%[1]Synthetic image
% ImagName = 'Real2D';       %%%%%[2]Real data
% ImagName = 'NoDrug2D';     %%%%%[3]Real data--No-drug
% ImagName = 'Drug2D';       %%%%%[4]Real data--drug

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im=double(imread([ImagName,'.tif']));    

%% %%%%%%%%%%%%%%%%%%%% compute convex hull 
Thresh=0.065; ConvInfo=ConvexHullCells(im,Thresh);

%% %%%%%%%%%%%%%%%%%%%% Compute structral tensor 
rho=10;  EigInfo = coherence_orientation(im,rho); 

%% %%%%%%%%%%%%%%%%%%%%%% display information
para.Step  =20; %%% intensity of orientation
para.scl   =10; %%% length of orientation
DisplayImage(im,EigInfo,ConvInfo,para);

