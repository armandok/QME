function M = quintessentialfactory(r)
% Returns a manifold structure to optimize over the (extended) reduced
% quintessential matrix manifold with the given rank=r.
%
% Some function definitions are similar to the stiefelfactory from Manopt
%

    assert(r >= 1, ...
        'The reduced quintessential matrix should have rank more than zero');
    assert(r <= 12, ...
        'The educed quintessential matrix matrix cannot have rank more than 12');
    
    
    M.name = @() sprintf('reduced quintessential manifold with rank= %d', r);
    
    M.dim = @() 12*r - 7;
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('quintessential.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(3);
    
    M.proj = @projection;
    function Up = projection(X, U)
        
        symXtU = 0.5*(X.'*U + U.'*X);
        Un = U - X * symXtU;
        
        XXt = X*X.';
        Proj = eye(4*r) - XXt;
        J = [0;0;0;1]*[0,0,0,1];
        G = kron(eye(r), J);
        
        c_nom = sum(dot(X(4:4:4*r,:),Un(4:4:4*r,:)),'all');
        c_denom = 1-norm(X(4:4:4*r,:).'*X(4:4:4*r,:),'fro')^2;
        c = c_nom/c_denom;
        if isnan(c) || isinf(c)
            c=0;
        end

        Up = Un - Proj * G * X * c;

    end
    
    M.tangent = M.proj;
    
    M.tangent2ambient_is_identity = true;
    M.tangent2ambient = @(X, U) U;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
    M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        GX = zeros(size(X));
        GH = zeros(size(X));
        GX(4:4:4*r,:)=X(4:4:4*r,:);
        GH(4:4:4*r,:)=H(4:4:4*r,:);
        
        Xtg = X.'*egrad;
        
        Xtg_sym = 0.5*(Xtg+Xtg.');
        
        MM = X(4:4:4*r,:).'*X(4:4:4*r,:);
        c_nom = sum(dot(MM, Xtg));
        c_denom = 1 - sum(dot(MM,MM));
        c = c_nom / c_denom;
        if isnan(c) || isinf(c)
            c=0;
        end
        
        XtGXdot = X(4:4:4*r,:).'*H(4:4:4*r,:);
        c_nom_dot = 2*sum(dot(XtGXdot, Xtg_sym)) ...
                    +sum(dot( MM, H.'*egrad + X.'*ehess ));
        c_denom_dot = -4*sum(dot(MM,XtGXdot));

        c_dot = c_nom_dot/c_denom - c_denom_dot*c_nom/c_denom^2;
        if isnan(c_dot) || isinf(c_dot)
            c_dot=0;
        end
        
        to_project = ehess-H*Xtg_sym-c_dot*GX-c*(GH-H*MM);
        rhess = projection(X, to_project);
    end


    function V = to_stiefel(Y)
        V = zeros(4*r,3);
        for idx = 1:r
            V(4*idx-3:4*idx, 1) = Y(1:4,idx);
            V(4*idx-3:4*idx, 2) = Y(5:8,idx);
            V(4*idx-3:4*idx, 3) = Y(9:12,idx);
        end
    end
    
    function Y = to_burer_monteiro(V)
        Y = zeros(12,r);
        for idx = 1:r
            Y(1:4,idx)  = V(4*idx-3:4*idx, 1);
            Y(5:8,idx)  = V(4*idx-3:4*idx, 2);
            Y(9:12,idx) = V(4*idx-3:4*idx, 3);
        end
    end
    
    M.retr = @retraction;
    function Y = retraction(X, U, t)
        if nargin < 3
            Y = retraction_impl_newton(X + U);
        else
            Y = retraction_impl_newton(X + t*U);
        end
    end
    

    function X = retraction_heuristic(Xp)
        X = zeros(size(Xp));
        
        T = Xp(4:4:4*r,:);
        nrm = norm(T,'fro');

        X(4:4:4*r,:) = Xp(4:4:4*r,:)/nrm;
        T = X(4:4:4*r,:);

        K = eye(3) - T.'*T;
        G = Xp(setdiff(1:4*r,4:4:4*r),:); % rows that are not in T

        H = G.'*G;
        

        K_sqr = real(sqrtm(K)); %K^(0.5);

        
        [U_Z,S_Z,~] = svd(K_sqr*H*K_sqr);
        X(setdiff(1:4*r,4:4:4*r),:) = G * K_sqr * U_Z * pinv(real(sqrt(S_Z))) * U_Z.' * K_sqr;
    end


    function [Xp] = retraction_impl_newton(Xp)
        delta_x = 1;
        while norm(delta_x) > 1e-4
            J = get_constraints_jacobian(Xp);
            f = get_constraints_vec(Xp);

            y = (J*J.')\f;
            delta_x = - J.'*y;
            Xp(:,1) = Xp(:,1) + delta_x(1:4*r);
            Xp(:,2) = Xp(:,2) + delta_x(4*r+1:8*r);
            Xp(:,3) = Xp(:,3) + delta_x(8*r+1:12*r);
        end
    end

    function [J] = get_constraints_jacobian(Xp)
        J = zeros(7,12*r);
        J(1, 1:4*r)      = 2*Xp(:,1);
        J(2, 4*r+1:8*r)  = 2*Xp(:,2);
        J(3, 8*r+1:12*r) = 2*Xp(:,3);

        J(4, 1:4*r)      = Xp(:,2);
        J(4, 4*r+1:8*r)  = Xp(:,1);
        
        J(5, 4*r+1:8*r)   = Xp(:,3);
        J(5, 8*r+1:12*r)  = Xp(:,2);
        
        J(6, 1:4*r)       = Xp(:,3);
        J(6, 8*r+1:12*r)  = Xp(:,1);
        
        J(7, 4:4:4*r)       = 2*Xp(4:4:4*r,1);
        J(7, 4*r+4:4:8*r)   = 2*Xp(4:4:4*r,2);
        J(7, 8*r+4:4:12*r)  = 2*Xp(4:4:4*r,3);
    end
    
    function [f] = get_constraints_vec(Xp)
        f = zeros(7,1);
        
        f(1) = Xp(:,1).'*Xp(:,1)-1;
        f(2) = Xp(:,2).'*Xp(:,2)-1;
        f(3) = Xp(:,3).'*Xp(:,3)-1;

        f(4) = Xp(:,1).'*Xp(:,2);

        f(5) = Xp(:,2).'*Xp(:,3);

        f(6) = Xp(:,3).'*Xp(:,1);

        f(7) = norm(Xp(4:4:4*r,:),'fro')^2-1;
    end

    M.exp = @(X, U, t) error('factorized_essential.exp not implemented yet.');

    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @() retraction_heuristic(randn(4*r, 3));
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(4*r, 3));
        U = U / norm(U(:));
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(4*r, 3);
    
    M.transp = @(x1, x2, d) projection(x2, d);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [4*r, 3]);
    M.vecmatareisometries = @() true;
    
end