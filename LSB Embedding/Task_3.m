close all
clear all

% Load the cover color image
I = imread('lena256.png');

% Load the secret image that will be hidden into I
Secret_img = rgb2gray( imread('che.jpg') );

% Convert the secret image into a binary image
bw_secret = im2bw(Secret_img);
figure('name','Black&white secret message'), imshow(bw_secret)

% Split the R,G,B color channels
I_red = I(:,:,1);
I_green = I(:,:,2);
I_blue = I(:,:,3);

% Hide the binary image in the LSB of the host image's green channel
[rows, cols] = size(I_green);

% We want to embed this message in the LSB plane 
planes = zeros(rows,cols,8);
for i=1:8
    planes(:,:,i) = bitget(I_green,i);
end

planes(:,:,1) = bw_secret;

% Reconstruct
for i=1:size(planes,1)
    for j=1:size(planes,2)
        binaryPixel = planes(i,j,:);
        binaryPixel = reshape(binaryPixel,1,length(binaryPixel),1);
        I_green_mod(i,j) = bi2de(binaryPixel);
    end
end
I_green_mod = uint8(I_green_mod);

% Merge the RGB color channel (G channel hides the secret image)
I_mod = cat(3,I_red,I_green_mod,I_blue);

% Show original and modified images
figure, imshow(I)
figure, imshow(I_mod)


%-------------------------------------------------------------------------%
% 2. RECOVER THE HIDDEN MESSAGE                                           %
%-------------------------------------------------------------------------%

% Extract the green channel 
I_green_rec = I_mod(:,:,2);

% LSB of the green channel and secret message
LSBplane = bitget(I_green_rec,1);
secret_rec = uint8( LSBplane );
figure, imshow(255*secret_rec)

% Compute the number of errors in reconstruction
errors = sum( sum( xor( bw_secret, secret_rec) ) )

imwrite(I_mod, 'lena256_stego.tif','tif','Compression','none');


% RedChannel
pixelwise = imabsdiff(I_red,I_mod(:,:,1));
distortion = sum(sum(pixelwise.^2))
PSNR = psnr(I_red,I_mod(:,:,1))
StructuralSimilarity = SSIM(I_red,I_mod(:,:,1))


% GreenChannel
pixelwise = imabsdiff(I_green,I_mod(:,:,2));
distortion = sum(sum(pixelwise.^2))
PSNR = psnr(I_green,I_mod(:,:,2))
StructuralSimilarity = SSIM(I_green,I_mod(:,:,2))



% Blue Channel
pixelwise = imabsdiff(I_blue,I_mod(:,:,3));
distortion = sum(sum(pixelwise.^2))
PSNR = psnr(I_blue,I_mod(:,:,3))
StructuralSimilarity = SSIM(I_blue,I_mod(:,:,3))

% Color difference
ColorDifference = sum(sum(sqrt(double((I_green - I_green_mod).^2 + (I_red - I_red).^2 + (I_blue - I_blue).^2))));


%Replace all the element related to green chanel to obtain the result for
%the other color.

