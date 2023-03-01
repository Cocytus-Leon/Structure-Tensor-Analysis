function loc= VectorLocation3D(nx,ny,nz,indx,indy,indz)
% nx, ny, nz -- the size of 3D matrix
% indx,indy,indz  -- the location of point
% loc  -- the order of the point

loc= (indz-1)*nx*ny + (indy-1)*nx + indx;