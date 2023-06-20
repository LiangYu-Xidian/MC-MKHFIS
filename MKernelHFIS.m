function [predict_label,score_s,omega_train,Alpha] = MKernelHFIS(X_train,Y_train,X_test,options)
%
%
% options.lambda Regularization coefficient
% options.k: number of fuzzy rules
% options.h: adjustable parameter for fcm used for generating antecedent
%             parameters.
% options.type: type of kernel function
% options.gamma: width of kernelq
%
%
seed = 12345678;
rand('seed', seed);
neighbor=35;
lamda=2^-5;
y_pred_test=[];
[n_tr,~] = size(X_train);


%if-parts
[v_train,b_train] = gene_ante_fcm(X_train,options);
[v_test,b_test] = gene_ante_fcm(X_test,options);

G_train = calc_x_g(X_train,v_train,b_train);
G_test = calc_x_g(X_test,v_test,b_test);


K_train = kernel_function_m(X_train,X_train,options.gamma,options.type);
K_test = kernel_function_m(X_test,X_train,options.gamma,options.type);


K_train1 = K_train(:,:, 1);
K_train2 = K_train(:,:, 1);

kernel_weights = cka_kernels_weights(K_train, Y_train, 1);

% kernel_weights

all_weight = sum(kernel_weights);
for i = 1:length(kernel_weights)
   kernel_weights(i) = kernel_weights(i)/all_weight; 
end

train_Kernel = combine_kernels(kernel_weights, K_train);
test_Kernel = combine_kernels(kernel_weights, K_test);

omega_train = zeros(n_tr,n_tr);
omega_test = zeros(size(X_test,1),n_tr);
Ones_Ma = ones(n_tr,n_tr);
Ones_Mb = ones(size(X_test,1),n_tr);
%computing omega matrix

for i=1:options.k
	a = G_train(:,i);
	A = a*a';
	
	b = G_test(:,i);
	B = b*a';      
	
	omega_train_i = (train_Kernel + Ones_Ma).*A;
	omega_train = omega_train + omega_train_i;
	
	omega_test_i = (test_Kernel + Ones_Mb).*B;
	omega_test = omega_test + omega_test_i;
	
end

omega_train = X_(omega_train);
omega_test = X_(omega_test);
omega_train(isnan(omega_train)) = 0;
omega_test(isnan(omega_test)) = 0;

%training
Alpha = (omega_train + options.lambda*eye(n_tr))\Y_train;

%predicting
score_s = omega_test*Alpha;
predict_label = sign(score_s);

end
