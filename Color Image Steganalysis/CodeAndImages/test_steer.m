close all;
clear all;
clc

 % features extraction  enriching the Color Rich Model
 %
 % Hasan Abdulrahman, Marc Chaumont, Philippe Montesinos, and Baptiste Magnier, 
 % "Color Image Steganalysis Based on Steerable Gaussian Filters Bank," 
 % IH&MMSec'2016, in Proceedings of the 4th ACM workshop on Information Hiding and Multimedia Security, 
 % Vigo, Galicia, Spain, 6 pages, June 20-22, 2016
 %
 % these features correspond to :
%  MaxI_R : gradient magnitude image of the red channel
%  MaxI_G : gradient magnitude image of the green channel
%  MaxI_B : gradient magnitude image of the blue channel
%  R_theta_90 : the tangent derivative of the red channel (perpendicular to the gadient of the red channel)
%  G_theta_90 : the tangent derivative of the green channel (perpendicular to the gadient of the green channel)
%  B_theta_90 : the tangent derivative of the blue channel (perpendicular to the gadient of the blue channel)
%
% note that gradient angle values are in : MaxI_angle_R, MaxI_angle_G and MaxI_angle_B

 delta_theta = 10;
 sigma = 0.7; 
 
 if (rem(180, delta_theta) > 0)
     msg = '360 must be divisible by delta_theta';
     error(msg)     
 end


XX=imread('LakeLSB.png');

 S = size(XX);
 if ((length(S) == 3 ) && (S(3)==3))
     disp('color image')     
 else
     msg = 'The input image is not a color image of 3 channels';
     size(XX)
     error(msg)     
 end
 
%%%%%%%
tic
theta = [0:delta_theta:(180-delta_theta)];
  
 
steer_Filter = cell(1,size(theta,2));
steer_Image_out_R = cell(1,size(theta,2));
steer_Image_out_G = cell(1,size(theta,2));
steer_Image_out_B = cell(1,size(theta,2));

R = XX(:,:,1); % red
G = XX(:,:,2); % green
B = XX(:,:,3); % blue
[nrows,ncols] = size(R)
  

for i = [1:length(theta)]
    [steer_Image_out_R{1,i},H] = steerGauss2(R,theta(i),sigma,true);
    [steer_Image_out_G{1,i},H] = steerGauss2(G,theta(i),sigma,true);
    [steer_Image_out_B{1,i},H] = steerGauss2(B,theta(i),sigma,true);
    filters{i} = H;
    %pause(0.1);
end


  
[MaxI_R MaxI_angle_R  Min_R ]= steer_max(steer_Image_out_R);
 R_theta_90 = value_of_theta(steer_Image_out_R, MaxI_angle_R, delta_theta);
     
[MaxI_G, MaxI_angle_G,  Min_G]= steer_max(steer_Image_out_G);
 G_theta_90 = value_of_theta(steer_Image_out_G, MaxI_angle_G, delta_theta);
    
[MaxI_B  MaxI_angle_B  Min_B ]= steer_max(steer_Image_out_B);
 B_theta_90 = value_of_theta(steer_Image_out_B, MaxI_angle_B, delta_theta);
  toc
  %%%%%%%%%%%%%%%%%%
  
  figure,
  subplot(3,4,1), imagesc(R), colormap(gray),axis off , title('red');
  subplot(3,4,2), imagesc(MaxI_R), colormap(gray),axis off , title('|\nabla R|');
  subplot(3,4,3), imagesc(MaxI_angle_R), colormap(gray),axis off , title('\theta_R');
  subplot(3,4,4), imagesc(R_theta_90), colormap(gray),axis off , title('R_{\theta+/-90}');
  
  subplot(3,4,5), imagesc(G), colormap(gray),axis off , title('green');
  subplot(3,4,6), imagesc(MaxI_G), colormap(gray),axis off , title('|\nabla G|');
  subplot(3,4,7), imagesc(MaxI_angle_G), colormap(gray),axis off , title('\theta_G');
  subplot(3,4,8), imagesc(G_theta_90), colormap(gray),axis off , title('G_{\theta+/-90}');
  
  subplot(3,4,9), imagesc(B), colormap(gray),axis off , title('blue');
  subplot(3,4,10), imagesc(MaxI_B), colormap(gray),axis off , title('|\nabla B|');
  subplot(3,4,11), imagesc(MaxI_angle_B), colormap(gray),axis off , title('\theta_B');
  subplot(3,4,12), imagesc(B_theta_90), colormap(gray),axis off , title('B_{\theta+/-90}');
  
% color: gradient and its orientation (optional)

[nrows,ncols] = size(MaxI_R);

gradient_c(:,:,1) = MaxI_R;
gradient_c(:,:,2)= MaxI_G;
gradient_c(:,:,3)= MaxI_B;

angle_c(:,:,1)= MaxI_angle_B;
angle_c(:,:,2)= MaxI_angle_G;
angle_c(:,:,3)= MaxI_angle_B;

for i = 2:3 
    for x = 1:nrows
        for y = 1:ncols
            if (gradient_c(x,y,1)<gradient_c(x,y,i) )
                gradient_c(x,y,1)=gradient_c(x,y,i);
                angle_c(x,y,1)= angle_c(x,y,i);
            end
        end
    end
end


% imwrite(uint8(max_grad),'max_grad_color.png','png');

figure,
  subplot(1,3,1), imagesc(XX), colormap(gray),axis off , title('Color image I');
  subplot(1,3,2), imagesc(gradient_c(:,:,1)), colormap(gray),axis off , title('|\nabla I|');
  subplot(1,3,3), imagesc(angle_c(:,:,1)), colormap(gray),axis off , title('\theta');