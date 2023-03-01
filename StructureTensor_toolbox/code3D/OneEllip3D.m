function FVal = OneEllip3D(A,Xp,Yp,Zp)
% % generate an 3D ellipsoid.
% % i.e., \|x-x0\|_A=1
% %  Written by Dr. Wenxing Zhang, ITAV, Toulouse, France, 2014.4.
% %        wenxing84@gmail.com.

FVal = (A(1,1)*Xp+A(2,1)*Yp+A(3,1)*Zp).*Xp ...
     + (A(1,2)*Xp+A(2,2)*Yp+A(3,2)*Zp).*Yp ...
     + (A(1,3)*Xp+A(2,3)*Yp+A(3,3)*Zp).*Zp;
% FVal = (A(1,1)*Xp+A(2,1)*Yp+A(3,1)*Zp).*Xp+(A(1,2)*Xp+A(2,2)*Yp+A(3,2)*Zp).*Yp + (A(1,3)*Xp+A(2,3)*Yp+A(3,3)*Zp).*Zp;
end