function [MaxI, I1, MinI ]= steer_max( steer1)
% steer1{1} drivative images
%% MaxI is the Maximum value from steerable filter images
%% I1 is the angle of maximum value
%% I2 is the angle of Minimum value.

[nrows,ncols] = size(steer1{1});
L = length(steer1); % nb de rotations
imAbs = zeros(nrows,ncols,L);
MaxI= zeros(nrows,ncols);
I1 = ones(nrows,ncols);
MinI= zeros(nrows,ncols);
for i = 1:L 
        imAbs(:,:,i) = abs(steer1{i}); % valeur absolue de la derivee 

%         if i ==5
%             figure ,
%             image(imAbs(:,:,i)), title('imAbs(:,:,i)');
%         end
end

MaxI=imAbs(:,:,1);
MinI=imAbs(:,:,1);

for i = 2:L 
    for x = 1:nrows
        for y = 1:ncols
            if (MaxI(x,y)<imAbs(x,y,i) )
                MaxI(x,y)=imAbs(x,y,i);
                I1(x,y) = i;
            end
            if (MinI(x,y)>imAbs(x,y,i) )  % optional
                MinI(x,y)=imAbs(x,y,i);
            end
        end
    end
end




% figure ,
% imagesc(MaxI), colormap(gray), title('MaxI dans steer_max');
% figure ,
% imagesc(I1), colormap(gray), title('I1 dans steer_max');
%  figure (2);
%  imagesc(MinI); axis image; colormap(gray);
%  figure (3);
%  imagesc(MaxI_MinI); axis image; colormap(gray);