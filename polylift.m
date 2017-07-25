function P = polylift(X,d,c)
%POLYLIFT create exact polynomial feature mapping
%cooresponding to polynomial kernel matrix (X'*X+c).^d.
%Each column of the input matrix X is lifted to all
%monomials of degree d.
%c = 0 is homogeneous mapping.
%c ~= 0 is inhomogeous mapping.
%Only supports d = 1,2, or 3.
%
%Example:
% X = randn(20,100);
% P = polylift(X,3,1);
% K = (X'*X+1).^3;
% norm(K-P'*P,'fro')

if d > 3
    error('polylift.m: d>3 not supported.');
end

[n,s] = size(X);
if c ~= 0  %inhomogeneous case
    X = [c*ones(1,s); X]; %augment with constant row
    n = n+1;
end

P = zeros(nchoosek(n+d-1,d),s);
switch d
    case 1
        P = X;
    case 2
        c1 = sqrt(2);
        trimask = logical(triu(ones(n),1));
        for j = 1:s
            x = X(:,j);
            x2 = x*x';
            x22 = diag(x2);
            x21 = x2(trimask);
            P(:,j) = [c1*x21(:);x22(:)];
        end
        
    case 3
        cube1 = reshape(kron((1:n)' * ones(1,n), ones(n,1)), [n, n, n]);
        cube2 = permute(cube1,[2 1 3]);
        cube3 = permute(cube1,[3 1 2]);
        halfspace = (cube1 >= cube2) & (cube3 >= cube2) & (cube3 >= cube1);
        mask3_3 = (cube1 == cube2) & (cube2 == cube3) & (cube1 == cube3);
        mask3_111 = (cube1 ~= cube2) & (cube2 ~= cube3) & (cube3 ~= cube1);
        mask3_111(~halfspace) = false;
        mask3_21 = (cube1 == cube2) | (cube2 == cube3) | (cube1 == cube3);
        mask3_21(mask3_3) = false;
        mask3_21(~halfspace) = false;

        c3_111 = sqrt(6);
        c3_21 = sqrt(3);

        for j = 1:s
            x = X(:,j);
            x3 = reshape(kron(x * x', x), [n, n, n]);
            x3_111 = x3(mask3_111);
            x3_21 = x3(mask3_21);
            x3_3 = x3(mask3_3);
            P(:,j) = [c3_111*x3_111(:);c3_21*x3_21(:);x3_3(:)];
        end
end

end

