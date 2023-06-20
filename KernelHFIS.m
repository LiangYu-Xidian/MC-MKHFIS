function [predict_label,score_s,omega_train, omega_test ,Alpha, train_Kernel, test_Kernel] = KernelHFIS(X_train,Y_train,X_test,options)
% Higher-order Fuzzy Inference Systems with Kernel
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
y_pred_test=[];
[n_tr,~] = size(X_train);


%if-parts
[v_train,b_train] = gene_ante_fcm(X_train,options);
[v_test,b_test] = gene_ante_fcm(X_test,options);

G_train = calc_x_g(X_train,v_train,b_train);
G_test = calc_x_g(X_test,v_test,b_test);



train_Kernel = kernel_function(X_train,X_train,options.gamma,options.type);
test_Kernel = kernel_function(X_test,X_train,options.gamma,options.type);


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
omega_train(isnan(omega_train)) = 0;
omega_test(isnan(omega_test)) = 0;
omega_train = X_(omega_train);
omega_test = X_(omega_test);


%training
Alpha = (omega_train + options.lambda*eye(n_tr))\Y_train;

%predicting
score_s = omega_test*Alpha;
predict_label = sign(score_s);

end

function k= kernel_function(X,Y,gamma,type)

if strcmp(type, 'rbf')
	k = kernel_RBF(X,Y,gamma);
elseif strcmp(type,'lap')
	k = kernel_Laplace(X,Y,gamma);
elseif strcmp(type,'sig')
	k = kernel_sig(X,Y,gamma);
elseif strcmp(type,'Poly')
	k = kernel_Polynomial(X,Y,gamma);
end


end

%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma);
end


%sig kernel function
function k = kernel_sig(X,Y,gamma)
	coef = 0.01;
	s = X*Y'*gamma + coef; 
	
	k = tanh(s);
end


%Polynomial kernel function
function k = kernel_Polynomial(X,Y,gamma)
	coef = 0.01;
	d=3;
	k =  (X*Y'*gamma + coef).^d; 
end


%Laplace kernel function
function k = kernel_Laplace(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r = sqrt(r2);
	k = exp(-r*gamma);
end