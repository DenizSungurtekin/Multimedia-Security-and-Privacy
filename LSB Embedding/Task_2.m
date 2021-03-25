close all
clear all

% Load the cover image and convert it to grayscale
I = imread('lena256.png');
G = rgb2gray(I);
[rows, cols] = size(G);

%-------------------------------------------------------------------------%
% 1. EMBED A SECRET MESSAGE INTO THE LSB PLANE                            %
%-------------------------------------------------------------------------%

% We want to embed this message in the LSB plane 
planes = zeros(rows,cols,8);
for i=1:8
    planes(:,:,i) = bitget(G,i);
end
LSBplane = bitget(G,1);

% Embed a message of length L
text_message = repmat('Help me ' , [1,100]) ;
binary_msg_string = dec2bin( text_message);

% a trick to convert the string to an array
binary_msg = binary_msg_string(:)-'0';

L = length(binary_msg);

% We have to select L random positions (i,j) inside the plane
key = 897613453483121;
rand('state', key);
embed_pos = randperm( numel( LSBplane ) );

for i=1:L
    LSBplane( embed_pos(i) ) = binary_msg(i);
end

% Put the modified plane back to its place
planes(:,:,1) = LSBplane;

% Reconstruct
for i=1:size(planes,1)
    for j=1:size(planes,2)
        binaryPixel = planes(i,j,:);
        binaryPixel = reshape(binaryPixel,1,length(binaryPixel),1);
        G_mod(i,j) = bi2de(binaryPixel);
    end
end
figure('name','Image with secret message'), imshow(uint8(G_mod))


%-------------------------------------------------------------------------%
% 2. RECOVER THE HIDDEN MESSAGE                                           %
%-------------------------------------------------------------------------%

% Suppose that the receiver knows the key and the message length L
key = 897613453483121;
rand('state', key);
embed_pos = randperm( numel( LSBplane ) );

% fast way to extract all the bits with just one line. One could
% equivalently use a FOR cycle as follows:
%
% for i=1:L
%   recovered_msg(i) = LSBplane_mod( embedpos(i) );
% end
LSBplane_mod = bitget(G_mod,1);
recovered_msg = LSBplane_mod( embed_pos(1:L) )';

% Compute the number of errors in reconstruction
errors = sum( xor( binary_msg, recovered_msg))

% Compare both images
pixelwise = imabsdiff(G,uint8(G_mod))
distortion = sum(sum(pixelwise.^2))
PSNR = psnr(G,uint8(G_mod))
StructuralSimilarity = SSIM(G,uint8(G_mod))