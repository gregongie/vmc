function X = lrmc_cvx(sampmask,samples)
%LRMC_CVX solves nuclear norm minimization problem:
%minize_X ||X||_* subject to X(sampmask) = samples;
%using CVX: http://cvxr.com/cvx/
[n,s] = size(sampmask);
cvx_begin %quiet
    variable X(n,s)
    minimize(norm_nuc(X))
    subject to 
        X(sampmask) == samples;
cvx_end
end