clc
clear 
close all
addpath ./code3D ./TestData
rand('seed',0); randn('state',0)
%% %%%%%%%%%%% Test images
ImName='Synth3D_A.tif';
% ImName='Synth3D_B.tif';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im=double(ReadImage3D(ImName))/255;  dlt=[1 1 1]; 

%% %%%%%%%%%%%%%%%%%%%%%%% compute convex hull
Thresh=0.1;  ConvInfo=ConvexHullCells3D(im,Thresh,dlt);  

%% %%%%%%%%%%%%%%%%%%%% Compute structral tensor 
rho=5;  EigInfo = coherence_orientation3D(im,rho,dlt); 

%% %%%%%%%%%%%%%%%%%%%%% display info.
para.Step  = 10; 
DisplayImage3D(im,EigInfo,ConvInfo,para)



