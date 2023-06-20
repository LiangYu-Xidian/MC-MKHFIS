function [predict_label,score_s,omega_train, Theta ,Alpha, obj] = MC_KernelHIFS(X_train,Y_train,X_test,options)
% options.sigma1: width of the mixture correntropy Gaussian function 1
% options.sigma2: width of the mixture correntropy Gaussian function 1
% options.delta: weight value of the Gaussian function 1
% options.iterMax: Algorithm iterations
%
% IFS
% options.lambda Regularization coefficient
% options.k: number of fuzzy rules
% options.h: adjustable parameter for fcm used for generating antecedent
%             parameters.
% options.type: type of kernel function
% options.gamma: width of kernel
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


K_train = kernel_function_m(X_train,X_train,options.gamma,options.type);
K_test = kernel_function_m(X_test,X_train,options.gamma,options.type);

kernel_weights = cka_kernels_weights(K_train, Y_train, 1);

all_weight = sum(kernel_weights);
for i = 1:length(kernel_weights);
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
% Alpha = 100 * ones(n_tr, 1);
% Alpha = randn(n_tr,1);
 Alpha =  zeros(n_tr, 1);
% Alpha = (omega_train + options.lambda*eye(n_tr))\Y_train;
obj = [];
for i = 1 : options.iterMax
    Temp = omega_train*Alpha;
    
    exp1 = diag(diag(kernel_RBF(Y_train, Temp, options.sigma1)));
    exp2 = diag(diag(kernel_RBF(Y_train, Temp, options.sigma2)));
    Theta = (options.delta / (0.5 * options.sigma1)) * exp1 ...
        + ((1 - options.delta) / (0.5 * options.sigma2)) * 1;



    Alpha = (options.lambda * inv(Theta) + omega_train)\Y_train;



end


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

%loss
function loss = vector_loss(x, y)
    loss = norm(x - y);
end