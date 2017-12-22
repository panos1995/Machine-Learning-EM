function [ pC ] = findprob( T,X )
%FINDPROB Summary of this function goes here
%   Detailed explanation goes here
[N D] = size(X); 

K = size(T, 2); 

Nk = zeros(1,K);  
for k=1:K
    % Data for the k class; 
    setNk = find(T(:,k)==1);
    Xk = X(setNk,:);
    Nk(k) = size(Xk,1);
end

pC =  Nk/N;  

end

