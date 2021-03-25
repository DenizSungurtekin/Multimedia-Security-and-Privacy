close all;
clear all;


% Load the cover image
I = imread('lena256.png');
figure('name','Original image'), imshow(I)

% Convert to grayscale [0,255] image
G = rgb2gray(I);
[rows, cols] = size(G);
figure('name','Grayscale image'), imshow(G)

%-------------------------------------------------------------------------%
% 1. SEPARATE EACH BITPLANE OF THE INPUT IMAGE                            %
%-------------------------------------------------------------------------%

% Initialise the cube container
planes = zeros(rows,cols,8);

% The slow way: convert each pixel of the image to its binary value, which
% is represented as a string. Place each element of the string in the right
% slice of the bitplane matrix
for i=1:size(G,1)
    for j=1:size(G,2)
        binaryPixel = dec2bin( G(i,j) );
        
        % the last element of the string goes in the last slice of the
        % matrix and so on
        k=0;
        for b=length(binaryPixel):-1:1
            planes(i,j,8-k) = str2double( binaryPixel(b) );
            k = k+1;
        end
        
    end
end
% Display each bitplane separately
for i=1:8
    figure('name',['Bitplane ' num2str(i)]), imshow(planes(:,:,i));
end


%-------------------------------------------------------------------------%
% 2. RECONSTRUCT THE INPUT IMAGE FROM BITPLANES                           %
%-------------------------------------------------------------------------%

G_rec = zeros(size(planes,1),size(planes,2));

for i=1:size(planes,1)
    for j=1:size(planes,2)
        
        % Select all the bits of the current pixel
        binaryPixel = planes(i,j,:);
        binaryPixel = reshape(binaryPixel,1,length(binaryPixel),1);
        
        % Order them from the LSB to the MSB
        binaryPixel = binaryPixel(end:-1:1);
        
        % Convert the array to the corresponding decimal value
        G_rec(i,j) = bi2de(binaryPixel);
    end
end
G_rec = uint8(G_rec);
figure('name','Original vs reconstructed'),
subplot(1,2,1),imshow(G);
subplot(1,2,2),imshow(G_rec);


%-------------------------------------------------------------------------%
% 3. MANIPULATE THE SUBPLANES 8,4,1 AND RECONSTRUCT THE INITIAL IMAGE     %
%-------------------------------------------------------------------------%

to_change = [8,4,1];

% For each bitplane to modify (separately)
for p=1:numel(to_change)
    
    % Set the chosen subplane to 0
    planes(:,:, to_change(p) ) = zeros(rows,cols);
    G_rec = zeros(size(planes,1),size(planes,2));
    
    % Reconstruct
    for i=1:size(planes,1)
        for j=1:size(planes,2)
            binaryPixel = planes(i,j,:);
            binaryPixel = reshape(binaryPixel,1,length(binaryPixel),1);
            binaryPixel = binaryPixel(end:-1:1);
            G_rec(i,j) = bi2de(binaryPixel);
        end
    end
    G_rec = uint8(G_rec);
    
    % Display
    figure('Name',['Bitplane' num2str( to_change(p) ) ' set to 0']),
    subplot(1,2,1),imshow(G), title('Original');
    subplot(1,2,2),imshow(G_rec), title('Modified');

end

