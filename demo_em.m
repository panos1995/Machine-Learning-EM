clear all; 
close all;
load mnist_all.mat;
K = 10;
T = []; 
X = [];
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
figure; 
hold on; 
for j=1:10

    s = ['train' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);   
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1; 
    X = [X; Xtmp]; 
    T = [T; Ttmp]; 
    
    s = ['test' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp]; 
   
    % plot some training data
    ind = randperm(size(Xtmp,1));
    for i=1:10
        subplot(10,10,10*(j-1)+i);     
        imagesc(reshape(Xtmp(ind(i),:),28,28)');
        axis off;
        colormap('gray');     
    end
 
end
X(X<3)=0;
X(X>0)=1;
Xtest(Xtest<3)=0;
Xtest(Xtest>0)=1;
[N, D] = size(X);

vertex_k = [1,2,4,8,16,32];
outputs=zeros(6,3);
Pc=findprob(T,X);%P of each category Summing at 1
[~, Ttrue] = max(TtestTrue,[],2); 
errors=zeros(1,size(vertex_k,2));
m_for_plot= zeros(32,784,10);               %gnwrizoume oti gia k=32 to kalutero
for i=1:size(vertex_k,2)
pi = ones(vertex_k(i),1)/vertex_k(i);
mu=(0.2).*rand(vertex_k(i),D)+0.4;%random_numbers between(0.4,0.6)
probtest=zeros(size(Xtest,1),10);
for c=1:10
setNk = find(T(:,c)==1);
Xk = X(setNk,:);
[G,M,P] = EM_train2(Xk,vertex_k(i),mu,pi);

if(vertex_k(i)==32)
m_for_plot(:,:,c)=M; % to 8elw gia to plot
end
probtest(:,c)=E_M_test(P,M,Pc(c),Xtest);
end

[~,Ttest] = max(probtest,[],2);
err = length(find(Ttest~=Ttrue))/sum(Ntest);
errors(i)=err;
 disp(['mixtures of K = ' num2str(vertex_k(i))])
 disp(['The error of the method is: ' num2str(err)])
end
[value,index]=min(errors);
disp(['the best method got K :  ' num2str(vertex_k(index))])
disp([' with error : ' num2str(value)])


