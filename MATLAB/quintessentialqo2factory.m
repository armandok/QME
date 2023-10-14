function M = quintessentialqo2factory()
% This is a 4 by 4 by 2 matrix manfiold such that the last columns of the
% two 4 by 4 matrices are perpendicular and both of these matrices are
% orthogonal (belong to O(4)).
%
% This manifold is used to perform local optimization on the signed
% quintessential manifold. We use it to refine the estimate of an essential
% matrix obtained from a semidefinite relaxation.
%
% This implementation is mostly borrowed from the stiefelfactory from
% Manopt


    n = 4;
    p = 4;
    k = 2;

    assert(n >= p, 'The dimension n must be larger than the dimension p.');

    M.name = @() sprintf('Quintessential qo2 manifold St(%d, %d)^%d', n, p, k);

    
    M.dim = @() k*(n*p - .5*p*(p+1));
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('quintessentialqo2.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(p*k);
    
    M.proj = @projection;
    function Up = projection(X, U)
        
        XtU = multiprod(multitransp(X), U);
        symXtU = multisym(XtU);
        Up = U - multiprod(X, symXtU);
        
        % Second step
        c = .5 * (X(:,4,1).'*Up(:,4,2) + X(:,4,2).'*Up(:,4,1));
        
        q_r = X(:,1:3,1).'*X(:,4,2);
        q_l = X(:,1:3,2).'*X(:,4,1);

        K1 = zeros(4);
        K2 = zeros(4);

        K1(1:3,4) = q_r;
        K1(4,1:3) = -q_r;

        K2(1:3,4) = q_l;
        K2(4,1:3) = -q_l;

        Up(:,:,1) = Up(:,:,1) - c * X(:,:,1) * K1;
        Up(:,:,2) = Up(:,:,2) - c * X(:,:,2) * K2;
    end
    
    M.tangent = M.proj;
    
    M.tangent2ambient_is_identity = true;
    M.tangent2ambient = @(X, U) U;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
    M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        XtG = multiprod(multitransp(X), egrad);
        symXtG = multisym(XtG);
        HsymXtG = multiprod(H, symXtG);
        
        Up = egrad - multiprod(X, symXtG);
        
        % Second step
        c = .5 * (X(:,4,1).'*Up(:,4,2) + X(:,4,2).'*Up(:,4,1));
        
        q_r = X(:,1:3,1).'*X(:,4,2);
        q_l = X(:,1:3,2).'*X(:,4,1);

        K1 = zeros(4);
        K2 = zeros(4);
        K1(1:3,4) = q_r;
        K1(4,1:3) = -q_r;
        K2(1:3,4) = q_l;
        K2(4,1:3) = -q_l;

        qd_r = H(:,1:3,1).'*X(:,4,2);
        qd_l = H(:,1:3,2).'*X(:,4,1);

        Kd1 = zeros(4);
        Kd2 = zeros(4);
        Kd1(1:3,4) = qd_r;
        Kd1(4,1:3) = -qd_r;
        Kd2(1:3,4) = qd_l;
        Kd2(4,1:3) = -qd_l;

        adjustment = zeros(size(ehess));

        adjustment(:,:,1) = c * (H(:,:,1) * K1 + X(:,:,1)*Kd1);
        adjustment(:,:,2) = c * (H(:,:,2) * K2 + X(:,:,2)*Kd2);

        Wp = ehess - HsymXtG;
        c_dot_1 = .5 * (H(:,4,1).'*Up(:,4,2) + X(:,4,2).'*Wp(:,4,1));
        c_dot_2 = .5 * (H(:,4,2).'*Up(:,4,1) + X(:,4,1).'*Wp(:,4,2));

        adjustment(:,:,1) = adjustment(:,:,1) + c_dot_1 * X(:,:,1) * K1;
        adjustment(:,:,2) = adjustment(:,:,2) + c_dot_2 * X(:,:,2) * K2;
        
        rhess = projection(X, ehess - adjustment - HsymXtG);
    end
    
    function Y = retraction(X, U, t)
        if nargin < 3
            Y = retraction_impl(X + U);
        else
            Y = retraction_impl(X + t*U);
        end
    end

    function Xp = retraction_impl(X)
        MM = zeros(4,2);
        MM(:,1) = X(:,4,1);
        MM(:,2) = X(:,4,2);
        [U,~,V] = svd(MM,'econ');

        MM = U*V.';

        X(:,4,1) = MM(:,1);
        X(:,4,2) = MM(:,2);

        [Q1, ~] = qr_unique(fliplr(X(:,:,1)));
        [Q2, ~] = qr_unique(fliplr(X(:,:,2)));

        Xp(:,:,1) = fliplr(Q1);
        Xp(:,:,2) = fliplr(Q2);
    end
    
    M.retr = @retraction;
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @() retraction_impl(randn(n, p, k));
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(n, p, k));
        U = U / norm(U(:));
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, p, k);
    
    M.transp = @(x1, x2, d) projection(x2, d);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [n, p, k]);
    M.vecmatareisometries = @() true;
    
end
