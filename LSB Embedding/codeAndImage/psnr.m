function p = psnr(A,B)

[m,n,c] = size(A);

mse=0;

% average PSNR if image is not single channel
for i=1:c
   tmp = (double(A(:,:,i)) - double(B(:,:,i))) .^ 2;
   mse = mse + sum(sum(tmp))/(m*n*c);
end

p = 10 * log10( 255^2 / mse);

