
close all

%-------------------------------------------------------------------------%
% 1. NOT ROBUST TO TYPICAL IMAGE PROCESSING                               %
%-------------------------------------------------------------------------%

I_stego = imread('lena256_stego.tif');

% One line to check whether message is still there
figure('name','Secret image'), 
subplot(1,2,1), imshow( I_stego ), title('Stego Image');
subplot(1,2,2), imshow( 255 * bitget(I_stego(:,:,2),1) );

% Crop the image: the message survives (because it was not scrambled!)
I_crop = imcrop(I_stego,[1,1,256,128]);

figure('name','Crop'), 
subplot(1,3,1), imshow( I_stego ), title('Stego Image');
subplot(1,3,2), imshow( I_crop ), title('Crop');
subplot(1,3,3), imshow( 255 * bitget(I_crop(:,:,2),1) ), title('Message');

% Add noise to the image
I_noise = imnoise(I_stego,'salt & pepper', 0.3);

figure('name','Salt & pepper noise'), 
subplot(1,3,1), imshow( I_stego ), title('Stego Image');
subplot(1,3,2), imshow( I_noise ), title('Image with noise');
subplot(1,3,3), imshow( 255 * bitget(I_noise(:,:,2),1) ), title('Message');

% JPEG compression with maximum quality
imwrite(I_stego,'lena256_stego.jpg','jpeg','Quality',100);
I_jpeg = imread('lena256_stego.jpg');

figure('name','JPEG Compression'), 
subplot(1,3,1), imshow( I_stego ), title('Stego Image');
subplot(1,3,2), imshow( I_jpeg ), title('Compressed Image');
subplot(1,3,3), imshow( 255 * bitget(I_jpeg(:,:,2),1) ), title('Message');

% Histogram equalisation to enhance contrast (only green channel)
I_hist = histeq( I_stego(:,:,2) );

figure('name','Histogram equalisation'), 
subplot(1,3,1), imshow( I_stego ), title('Stego Image');
subplot(1,3,2), imshow( I_hist ), title('Contrast enhanced');
subplot(1,3,3), imshow( 255 * bitget(I_hist,1) ), title('Message');


%-------------------------------------------------------------------------%
% 2. NOT ROBUST TO STEGANALYSIS: THE SECRET IS NOT SO SECRET              %
%-------------------------------------------------------------------------%

% Traces in the histogram of the channel hiding the message
figure('name','Stego green histogram'), imhist(I_stego(:,:,2));

% Original histogram of the green channel
I = imread('lena256.png');
figure('name','Original green histogram'), imhist(I(:,:,2));


% See slides for more problems!




