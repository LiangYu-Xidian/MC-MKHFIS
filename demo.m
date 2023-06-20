clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; 




%  dataname = 'enhancer_Kmer4_training.csv';  options.lambda=0.1;options.h=1.6;options.type='MCKernel';options.iterMax= 10;options.k=5; options.gamma = [2^-2 2^-5 2^-8 2^-7];options.delta=0.1;options.sigma1=0.0625;options.sigma2=16;
%  dataname = 'enhancer_Kmer4_training.csv';  options.lambda=0.001;options.h=0.5;options.type='rbf';options.iterMax= 10;options.k=5; options.gamma = 2^-7;

%  dataname = 'enhancer_8_cell_Kmer4_new_training.csv';options.lambda=0.1;options.h=1;options.type='MCKernel';options.iterMax= 10;options.k=8; options.gamma = [2^0 2^1];options.delta=0.9;options.sigma1=2^-5;options.sigma2=2;
% dataname = 'enhancer_8_cell_Kmer4_new_training.csv';options.lambda=0.1;options.h=0.9;options.type='rbf';options.iterMax= 10;options.k=8; options.gamma = 2^-4;
  


 data = readmatrix(dataname); dataname
 [temp_n, temp_m] = size(data);
 x = data(:, 1 : temp_m - 1);
 y = data(:, temp_m);

% load(dataname); dataname

ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];auc=[];


X = x;
X(isnan(X)) = 0;

X = line_map(X);
KP = 1:1:length(y);
crossval_idx = crossvalind('Kfold',KP,nfolds);

X_Y_test_label=[];
X_Y_dis=[];
size(X)
S = [];
Y = [];
P_Y = [];
for fold=1:nfolds
 
 train_idx = find(crossval_idx~=fold);
 test_idx  = find(crossval_idx==fold);
 
 
 train_x = X(train_idx,:);
 train_y = y(train_idx,1);
 
 test_x = X(test_idx,:);
 test_y = y(test_idx,1);
 


 [predict_y,score_s,omega_train,Alpha] = MKernelHFIS(train_x,train_y,test_x,options);
 
%     [predict_y,score_s,omega_train, omega_test, Alpha, train_Kernel, test_Kernel] = KernelHFIS(train_x,train_y,test_x,options);
%   [predict_y,score_s,omega_train,Theta, Alpha] =  MC_MKernelHIFS(train_x,train_y,test_x,options);
% [predict_y,score_s] = tsk_fs(train_x,train_y,test_x,options);
%  [predict_y,score_s,omega_train,Alpha] = MKernelHFIS(train_x,train_y,test_x,options);
 [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc( predict_y,score_s, test_y );
 ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];
 acc = length(find(predict_y==test_y))/length(test_y)
 X_Y_test_label=[X_Y_test_label;test_y];
 X_Y_dis=[X_Y_dis;score_s];

		fprintf('- FOLD %d - ACC: %f \n', fold, ACC_i)
		%break;
end

 

mean_acc=mean(ACC)
mean_sn=mean(SN)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)
mean_auc=mean(auc)
 