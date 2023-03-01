% % Structure tensor based analysis of cells and nuclei organization
% % in tissues
% % 
% % This is the Toolbox for assessing the orientations and anisotropies
% % of cells and nuclei by structure tensor:
% %      J_\rho=K_\rho * (\nabla u  \nabla u ^T)
% % where J_\rho is the structure tensor, K_\rho is the Gaussian filter.
% % The square root of the ratio of both eigenvalues represents the 
% % anisotropy of cells and the eigenvector corresponding to
% % the largest eigenvalue is the orientation.
% % 
% % The codes can be used for both 2D and 3D analyses. 
% % demo2D.m  ---- a script to illustrate the method on 2D images
% % subroutines: 
% %    ConvexHullCells.m       --- computes the convexhull of
% %                                2D image content
% %    coherence_orientation.m --- Computes the structure tensor(J_\rho), 
% %                                the eigenvalues and eigenvectors;
% %    DisplayImage            --- displays original image, convexhull,
% %                                 orientations and anisotropy
% % 
% % demo3D.m  ---- a script to illustrate the method on 3D images
% % subroutines:
% %      ConvexHullCells3D        --- computes the convexhull of a 3D image
% %      coherence_orientation3D  --- Computes the structure tensor J_\rho, 
% %                                   the eigenvalues and eigenvectors
% %      DisplayImage3D           --- displays the original image, convexhull,
% %                                   orientations and anisotropy    
% %      
% % Test images in folder "TestImage" include:
% % 
% % 2D data:
% %     Synth2D  --- 512 x 512 synthetic image
% %     Real2D   --- 10240 x 1392 Real data
% %     NoDrug2D --- 10240 x 1392 Real data (with drug)
% %     Drug2D   --- 10240 x 1392 Real data (after drug)
% % 3D data:
% %     Synth3D_A--- 128 x 128 x 128 synthetic image, 
% %                  ratio of axes of cell: 5:5:2
% %     Synth3D_B--- 128 x 128 x 128 synthetic image,
% %                  ratio of axes of cell: 5:2:2
% % 
% % The codes can be downloaded at: 
% % http://www.math.univ-toulouse.fr/~weiss/
% % 
% % In case you use the results of this paper, please cite:
% % W.X. Zhang, J. Fehrenbach, A. Desmaison, V. Lobjois, B. Ducommun,
% % and P. Weiss, "Structure tensor based analysis of cells and nuclei
% % organization in tissues, preprint, July 2014
% % 
% % Feel free to contact us:
% % Email: pierre.armand.weiss@gmail.com; 
% %        wenxing84@gmail.com.
% %
% % If you use the test data, please cite:
% % ???
% % Date: 2014.7.18
