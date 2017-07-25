function K = rbf(X,X2,sigma)
%RBF Compute Gaussian RBF kernel matrix with bandwidth sigma
%Output is matrix K = k_sigma(X,X2) with entries given by
%    [K]_{i,j} = exp(-||x_i = x2_j||/(2*sigma^2))
%Call with X2 = [] to compute k_sigma(X,X) more efficiently
n1sq = sum(X.^2,1);
n1 = size(X,2);

if isempty(X2)
    D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
else
    n2sq = sum(X2.^2,1);
    n2 = size(X2,2);
    D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
end
K = exp(-D/(2*sigma^2));

end