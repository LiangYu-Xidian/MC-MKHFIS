clear
seed = 12345678;
rand('seed', seed);



%   dataname1 = 'enhancer_Kmer4_training.csv'; dataname2 = 'enhancer_Kmer4_testing.csv';  options.lambda=0.1;options.h=1.6;options.type='MCKernel';options.iterMax= 10;options.k=5; options.gamma = [2^-2 2^-5 2^-8 2^-7];options.delta=0.1;options.sigma1=0.0625;options.sigma2=16;
% dataname1 = 'enhancer_Kmer4_training.csv'; dataname2 = 'enhancer_Kmer4_testing.csv'; options.lambda=0.1;options.h=0.5;options.type='rbf';options.iterMax= 10;options.k=5; options.gamma = 2^-7;
%  dataname1 = 'enhancer_8_cell_Kmer4_new_training.csv'; dataname2 = 'enhancer_8_cell_Kmer4_new_testing.csv';  options.lambda=0.1;options.h=1;options.type='MCKernel';options.iterMax= 10;options.k=8; options.gamma = [2^0 2^1];options.delta=0.9;options.sigma1=2^-5;options.sigma2=2;
% dataname1 = 'enhancer_8_cell_Kmer4_new_training.csv'; dataname2 = 'enhancer_8_cell_Kmer4_new_testing.csv';options.lambda=0.1;options.h=0.9;options.type='rbf';options.iterMax= 10;options.k=8; options.gamma = 2^-4;


data1 = readmatrix(dataname1); dataname1
[temp_n1, temp_m1] = size(data1);
x1 = data1(:, 1 : temp_m1 - 1);
train_y = data1(:, temp_m1);
%% 


data2 = readmatrix(dataname2); dataname2
[temp_n2, temp_m2] = size(data2);
x2 = data2(:, 1 : temp_m2 - 1);
test_y = data2(:, temp_m2);

x = [x1; x2];
X = x;
X(isnan(X)) = 0;

X = line_map(X);
train_x = X(1 : temp_n1, : );
test_x = X(temp_n1 + 1 : temp_n1 + temp_n2 , : );


X_Y_test_label=[];
X_Y_dis=[];


 [predict_y,score_s,omega_train, omega_test, Alpha, train_Kernel, test_Kernel] = KernelHFIS(train_x,train_y,test_x,options);
%   [predict_y,score_s,omega_train,Alpha] = MKernelHFIS(train_x,train_y,test_x,options);
%  [predict_y,score_s,omega_train,Theta, Alpha, obj] =  MC_MKernelHIFS(train_x,train_y,test_x,options);
%  [predict_y,score_s] = tsk_fs(train_x,train_y,test_x,options);

 [ACC,SN,Spe,PE,NPV,F_score,MCC] = roc( predict_y, score_s, test_y );

 acc = length(find(predict_y==test_y))/length(test_y)
 X_Y_test_label=[X_Y_test_label;test_y];
 X_Y_dis=[X_Y_dis;score_s];

ACC
SN
Spe
MCC
 