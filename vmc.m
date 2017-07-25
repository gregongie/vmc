function [X,cost,update,error] = vmc(Xinit,sampmask,samples,options,Xtrue)
% VMC variety-based matrix completion algorithm.
% Inputs:
% Xinit     = initialization of matrix to be completed,
%             arranged in features-by-samples
% sampmask  = logical mask of sampling entries
% samples   = vector of sampled entries
% options   = struct of options (for detailed usage see below)
% Xtrue (optional) = ground truth matrix, 
%                    used to compute per iteration error.
% Outputs:
% X         = completed matrix
% cost      = vector of cost function value at each iteration
% update    = vector of change in relative NRMSE between iterates
% error     = vector of recovery error at each iteration
%             (non-empty only if Xtrue is passed)    
%
% VMC solves the optimization problem:
% minimize_X \|\phi(X)\|_{S_p}^p 
% subject to \|X(sampmask) - samples\|_2 <= epsilon
% where \|.\|_{S_p}^p is the Schatten-p matrix quasi-norm 0 <= p <= 1.
% \phi(X) is a polynomial feature map 
% epsilon is an estimate of the noise level.
%
% For more information, see the paper:
% G. Ongie, R. Willett, R. Nowak, L. Balzano.
% "Algebraic Variety Models for High-Rank Matrix Completion", in ICML 2017.
% Available online: https://arxiv.org/abs/1703.09631
% 
% Copyright 2017, Greg Ongie, University of Michgian
% E-mail questions and bug reports to: gongie@umich.edu
% Version 0.1, Updated 7/22/2017

% Set options:
%
% options.d = polynomial kernel degree (typically d=2 or 3) 
% options.c = polynomial kernel "bandwidth" parameter
% Kernel matrix has form K = (X'*X+c).^d
% Set c=0 for homogeneous kernel, c>0 for inhomogeneous kernel.
% An inhomogenous kernel may work better if data lies
% on a union of affine subspaces. 
% In this case, c=1 typically works well.
% Note: setting d=1,c=0 performs low-rank matrix completion
if ~isfield(options,'d')
    d = 2; %default to degree d=2 kernel (X'*X+c).^2
else
    d = options.d;
end
if ~isfield(options,'c')
    if d < Inf
        c = 0; %default to homogeneous poly kernel
    else
        c = 1; %Guassian RBF with bandwidth 1 
    end
else
    c = options.c;
end

% options.epsilon = noise level 
% smaller epsilon -> tighter data constraints
% Set epsilon = 0 for equality constraints: X(sampmask) = samples
% (this is the default behavior)
if ~isfield(options,'epsilon')
    epsilon = 0; %equality constraints
else
    epsilon = options.epsilon;   
end

% options.niter = max number of iterations
if ~isfield(options,'niter');
    niter = 5000;
else
    niter = options.niter; 
end

% options.gamma0 = inital smoothing parameter
% Set gamma0 = 0 to auto-initialize (experimental)
% Typically in the range gamma0 = 0.01--1.
% If eig or svd is throwing errors, try decreasing gamma0.
if ~isfield(options,'gamma0');
    gamma0 = 1; 
else
    gamma0 = options.gamma0;   
end
 
% options.gammamin = minimum allowed gamma.
if ~isfield(options,'gammamin');
    gammamin = 1e-16;
else
    gammamin = options.gammamin;
end

% options.eta = gamma decrease factor
% gamma update is min(gamma/eta,gammamin).
% Can choose larger eta for faster convergence
% but at the risk of converging to a local minimum
% Typical value is eta = 1.001--1.100;
if ~isfield(options,'eta');
    eta = 1.01;
else
    eta = options.eta;   
end

% options.p = Schatten-p penalty value
% Can choose 0 <= p <= 1
% p = 0 is the log-det penalty
% p = 1 is nuclear norm
% p = 0.5 works well for most settings
if ~isfield(options,'p');
    p = 0.5; %nonconvex Schatten-1/2 penalty
else
    p = options.p;   
end 

% options.exit_tol = convergence tolerance
% algorithm exits if relative change in NRMSE betwen iterates
% is less than this: ||X-Xold||_F/||Xold||_F < exit_tol
if ~isfield(options,'exit_tol');
    exit_tol = 1e-8;
else
    exit_tol = options.exit_tol;
end

% options.eigcomp = type of eigendecomposition/svd step
% for an n-by-s input matrix, let N = nchoosek(n+d-1,d);
% 'kernel-eig'   - construct s-by-s kernel matrix, full eigendecomp 
% 'kernel-rsvd'  - construct s-by-s kernel matrix, randomized truncated svd 
% 'lift-svd'     - construct N-by-s feature mapping, full (thin) svd
% 'lift-rsvd'    - construct N-by-s feature mapping, randomized truncated svd
% The 'lift-svd' and 'lift-rsvd' options may be more efficient
% if N < s, which can be the case for small n and small d.
if ~isfield(options,'eigcomp')
    [n,s] = size(Xinit);
    
    if d == Inf
        N = Inf;
    else
        if c==0
            N = nchoosek(n+d-1,d);
        else
            N = nchoosek(n+d,d);
        end
    end
    if N < s %use explicit lifting in polylift.m
        if N > 1000 %large scale
            eigcomp = 'lift-rsvd'; %use rsvd
        else
            eigcomp = 'lift-svd';  %use svd econ
        end
    else    %use kernel matrix
        if s > 1000 %large scale
            eigcomp = 'kernel-rsvd'; %use psd rsvd
        else
            eigcomp = 'kernel-eig';  %use eig
        end
    end
    fprintf('VMC set options.eigcomp=''%s'' based on problem size\n',eigcomp);
else
    eigcomp = options.eigcomp;
end

% options for rsvd if used:
% options.rmax = maximum number of eigenvalues to compute
% options.eigtol = threshold eigenvalues below ev(1)*eigtol
% The algorithm dynamically updates a rank threshold r
% based on eigtol. Set eigtol = 0 to disable this behavior.
% If you get the error "Input to SVD must not contain NaN or Inf."
% try increasing rmax and decreasing eigtol
if ~isfield(options,'rmax')
    [n,s] = size(Xinit);
    switch eigcomp
        case 'kernel-rsvd'
            rmax = min(1000,round(0.5*s));
            r = rmax;
        case 'lift-rsvd'
            [n,s] = size(Xinit);
            if c==0
                N = nchoosek(n+d-1,d);
            else
                N = nchoosek(n+d,d);
            end
            rmax = min(1000,round(0.5*min(N,s)));
            r = rmax;
    end
else
    rmax = options.rmax;
    r = rmax;
    rold = 0;
end
if ~isfield(options,'eigtol');
    eigtol = 1e-4; 
else
    eigtol = options.eigtol; 
end

% Determine cost function
if p == 0 
    Cp = 1/2;
    costfun = @(ev) 0.5*sum(log(ev+gammamin)); 
else
    Cp = p/2;
    costfun = @(ev) sum((ev+gammamin).^(p/2));
end

%initialize variables
%scale input according to max column 2-norm
scalefac = sqrt(max(sum(abs(Xinit).^2))); 
X = Xinit/scalefac; %normalize data;
samples = samples/scalefac;
epsilon = epsilon/scalefac;

Xold = X;
q = 1-(p/2); 

cost = [];
error = [];
update = [];
if nargin < 5
    havetruth = false;
else
    havetruth = true;
    Xtrue = Xtrue/scalefac;
end
if havetruth
    error(1) = norm(X-Xtrue,'fro')/norm(Xtrue,'fro');
end
% run alg
for i=1:niter;
    G = X'*X; %maintain gram matrix to save on matrix multiplys
    switch eigcomp
        case 'lift-svd'
            P = polylift(X,d,c);
            [~,D,V] = svd(P,'econ');
            ev = diag(D).^2;
        case 'lift-rsvd'
            P = polylift(X,d,c);
            [~,D,V] = rsvd(P,r);
            [ev,idx] = sort(diag(D).^2,'descend');
            V = V(:,idx);
        case 'kernel-eig'
            if d < Inf
                K = (G+c).^d;    %polynomial kernel
            else
                K = rbf(X,[],c); %Gaussian RBF kernel
            end
            [V,D] = eig(K);
            [ev,idx] = sort(abs(diag(D)),'descend');
            V = V(:,idx);
        case 'kernel-rsvd'
            if d < Inf
                K = (G+c).^d;    %polynomial kernel
            else
                K = rbf(X,[],c); %Gaussian RBF kernel
            end
            [V,D] = rsvdpsd(K,r,10,3);
            [ev,idx] = sort(abs(diag(D)),'descend');
            V = V(:,idx);
    end
    
    if i==1 %intialize gamma
        if gamma0 == 0 %auto-calibrate gamma
            gamma = 0.01*ev(1);
            %fprintf('VMC set gamma0=%1.2e\n',gamma);
        else
            gamma = gamma0;
        end
    end
    
    %compute weight matrix W
    switch eigcomp
        case {'kernel-eig','lift-svd'}
            evinv = (ev+gamma).^(-q);
            E = diag(evinv);
            W = V*E*V';
        case {'kernel-rsvd','lift-rsvd'}            
            evinv = (ev(1:r)+gamma).^(-q)-gamma^(-q);
            E = diag(evinv);
            W = V(:,1:r)*E*V(:,1:r)' + gamma^(-q)*eye(size(V,1));    
    end
              
    %projected gradient descent step
    if d == 1
        gradX = X*W;
    elseif d == 2
        gradX = 2*X*(W.*(G+c));
    elseif d > 2 && d < Inf
        gradX = d*X*(W.*((G+c).^(d-1)));
    elseif d == Inf
        gradX = X*(W.*rbf(X,[],c));
    end
       
    %gradient step
    tau = gamma^q;
    X = X - tau*gradX;
    if epsilon == 0 %project onto equality constraints       
        X(sampmask) = samples; 
    else  %project onto norm ball ||X(sampmask)-samples|| < epsilon
        nd = norm(X(sampmask)-samples);
        if nd > epsilon %need to project
            alpha = (epsilon/nd)^2;
            X(sampmask) = alpha*X(sampmask)+(1-alpha)*samples;
        end
        %otherwise keep X the same
    end

    %decrease smoothing parameter
    gamma = gamma/eta;
    gamma = max(gamma,gammamin);
    
    %update rank cutoff if using rsvd
    switch eigcomp
        case {'kernel-rsvd','lift-rsvd'}
            tmp = find(ev<eigtol*ev(1));
            if ~isempty(tmp)
                rhat = tmp(1)-1;
            else
                rhat = r; %min(r+1,rmax);
            end
            if rhat ~= r
                fprintf('rsvd rank cutoff updated to r=%d at iter=%d\n',r,i);
            end
            r = rhat;
    end
    
    % compute cost
    cost(i) = costfun(ev); 
    
    % compute error if have ground truth
    if havetruth
        error(i+1) = norm(X-Xtrue,'fro')/norm(Xtrue,'fro');
    end
    
    % check for convergence
    update(i) = norm(X-Xold,'fro')/norm(Xold,'fro');
    if( update(i) < exit_tol )
    fprintf('VMC reached exit tolerance at iter %d\n',i);
        break; 
    end    
    Xold = X;
end

X = scalefac*X; %rescale data;
end