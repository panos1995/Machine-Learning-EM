function [ G,M,P ] = EM_train2( Xin,K,mu,pi )
%EM_TRAIN Summary of this function goes here
%   Detailed explanation goes here

X=Xin;
[N,D]=size(X);
G=zeros(N,K);
M=mu;
P=pi;

%LET THE TRAIN BEGIN

prevcost=Inf;
it=0;
while 1
    M(M<0.0001)=0.0001;
    M(M>0.9999)= 0.9999;
    it=it+1;
    %   Å step G computation


helpx=1-X;
helpM=1-M;
f=zeros(N,K);

%     
          L=X*log(M.')+helpx*log(helpM.');
          
          phelp=transpose(repmat(log(P),1,size(f,1)));
          f=L+phelp;
          G=softmax(f);
  
     
    
   
    

    m=max(f,[],2);
    fhelp= f - repmat(m, 1, size(f,2));
    cost = sum(log(sum(exp(fhelp),2)) + m) ;
  
    %check for sugklisi
    %fprintf('Iteration %4d  Cost function %11.6f\n', it, cost); 
    if(it==1)
       prevcost=cost;
    end
    
    if(it~=1)
        if(cost-prevcost<0)
        disp('error at cost');
        break;
        end
    if(abs(cost-prevcost)<0.02)
%         disp('EM FINISHED iteration: ');
%         disp(it);
        break;
    else prevcost=cost;
    end
    end
    
      % Ì step
    
    
      fm=G'*X;
      per_col=transpose(sum(G,1));
      sumhelp=repmat(per_col,1,size(fm,2));
      M=fm./sumhelp;
%        
  
    %new p
    

      P=transpose(sum(G,1)./N);
    
    
end



end

