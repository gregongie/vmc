function X = lrmc_admm(Xinit,sampmask,samples,options)
%LRMC_ADMM solves regularized nuclear norm minimization problem
% minimize_X ||X||_* + (lambda/2)*||X(sampmask)-samples||^2;
% using the ADMM algorithm.
lambda = options.lambda; %regularization parameter
mu = options.mu; %ADMM parameter -- tune for fast convergence
niter = options.niter;

X = Xinit;
%error = zeros(1,niter);
%cost = zeros(1,niter);
L = zeros(size(X));
Y = zeros(size(X));
X0 = zeros(size(X));
X0(sampmask) = samples;
for i=1:niter
    %low-rank soft thresholding
    [U,S,V] = svd(X+L,'econ');
    s = diag(S);
    s = max(s-1/mu,0);
    Y = U*diag(s)*V';
    X = ((mu/lambda)*(Y-L)+X0)./((mu/lambda)+double(sampmask));
    L = L + X - Y;
    %error(i) = norm(X-Xtrue,'fro')/norm(Xtrue,'fro');
    %[~,S,~] = svd(X,'econ');
    %cost(i) = norm(diag(S),1)+0.5*lambda*norm(X(sampmask)-samples)^2;
end
end