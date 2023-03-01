function VisualData3D(data)
% % Illustrate the 3D isosurface of 3D data
% % Input: data -- n1 x n2 x n3 matrix
% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %        wenxing84@gmail.com.

%%%%%%%%% data is an nx*ny*nz matrix
data = smooth3(data,'gaussian',3);  %%%%% smooth the data
% patch(isocaps(data,0.5),'FaceColor','interp','EdgeColor','none');
p1 = patch(isosurface(data,0.5),'FaceColor','r','EdgeColor','none');
isonormals(data,p1)
view(3); 
axis equal
box on;
camlight left; 
lighting phong;

