function [  Ytest ] = E_M_test( P,M,Pc,Xtest )
%E_M_TEST Summary of this function goes here
%   Detailed explanation goes here

Ntest = size(Xtest,1); 
K = size(M,1);
 
% edw exoume gia ka8e Xtest ths pi8anotites gia ka8e Kc opou Kc einai
% plithos twn mixture.
logPr = Xtest*log(transpose(M)) + ((1-Xtest)*(log(transpose(1-M)))) +repmat(log(P'),Ntest,1);

% kai twra gia ka8e xtest pernoume to a8roisma ths seiras gia na vgaloume
% thn sunolikh timh tou xtest gia thn kathgoria c 

m = max(logPr,[],2);
Ytest = m + log( sum( exp( logPr  -  repmat(m, 1, K)   ), 2) ) + log(Pc);


end

