function K_train = kernel_function_m(X,Y,gamma_list,type)

K_train = [];
if strcmp(type, 'rbf')
	K_train(:,:,1) = kernel_RBF(X,Y,gamma_list(1));
elseif strcmp(type,'lap')
 	K_train(:,:,1) = kernel_Laplace(X,Y,gamma_list(2));
elseif strcmp(type,'sig')
	K_train(:,:,1) = kernel_sig(X,Y,gamma_list(3));
elseif strcmp(type,'Poly')
	K_train(:,:,1) = kernel_Polynomial(X,Y,gamma_list(4));
elseif strcmp(type, 'MCKernel')
    for i= 1 : length(gamma_list)
        K_train(:,:,i) = kernel_RBF(X,Y,gamma_list(i));
    end

%     for i= 1 : length(gamma_list)
%         K_train(:,:,i) = kernel_sig(X,Y,gamma_list(i));
%     end


%     for i= 1 : length(gamma_list)
%         K_train(:,:,i) = kernel_Polynomial(X,Y,gamma_list(i));
%     end


%        K_train(:,:,1) = kernel_RBF(X,Y,gamma_list(1));
%        K_train(:,:,2) = kernel_sig(X,Y,gamma_list(2));
%        K_train(:,:,3) = kernel_Polynomial(X,Y,gamma_list(3));

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