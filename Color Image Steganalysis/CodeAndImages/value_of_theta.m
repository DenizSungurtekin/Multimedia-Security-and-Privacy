function A1 = value_of_theta(steer1, A, delta_theta)
% orientation estimation

%A=theta_Q;
[u v ]=size(A)
% convert no. of theta to real theta
%%%%%%%%%%%%%%%%%%work with delta_theta
 B=(A-1)*delta_theta;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute Q2  
Q2= zeros(u,v);
for i =1 :u
    for j=1:v
        if (B(i,j)<90)
            Q2(i,j)=B(i,j)+90;
        else
            Q2(i,j)=B(i,j)-90;
        end
    end
end


%%%%%%%%%%%%%%%%%%  
 Q3 = fix((Q2/delta_theta)+1); 
 
%%%%%%%%%%%%%%%%%%%%%%%% derivative at the \theta +/- 90 orientation

for i =1:u
    for j=1:v
        A1(i,j)= steer1{Q3(i,j)}(i,j);
    end
end

% figure,
%  imagesc(theta_R); axis image; colormap(gray); 
% % 
% figure,
% imagesc(A1), colormap(gray); 
