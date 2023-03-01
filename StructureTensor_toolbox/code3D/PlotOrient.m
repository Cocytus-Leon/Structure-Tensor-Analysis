function PlotOrient(X,Y,Z,Tep1,Tep2,Tep3,CentVC,scl,color)
% % This code plot the orientation in 3D space with the 3D points
% % [X(CentVC),Y(CentVC),Z(CentVC)] 

% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %        wenxing84@gmail.com.

quiver3(Y(CentVC),X(CentVC),Z(CentVC),Tep1(CentVC),Tep2(CentVC),...
    Tep3(CentVC),scl,[color,'.'],'linewidth',3.5); 
quiver3(Y(CentVC),X(CentVC),Z(CentVC),-Tep1(CentVC),-Tep2(CentVC),...
    -Tep3(CentVC),scl,[color,'.'],'linewidth',3.5);