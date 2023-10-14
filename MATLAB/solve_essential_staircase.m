function [E, x] = solve_essential_staircase(F1, F2, varargin)

ivarargin=1;
verbose = false;
while(ivarargin<=length(varargin))
    switch(lower(varargin{ivarargin}))
        case 'verbose'
            ivarargin=ivarargin+1;
            verbose=varargin{ivarargin};
        otherwise
            error(['Argument ' varargin{ivarargin} ' not valid!'])
    end
    ivarargin=ivarargin+1;
end

positive_eigen_value_threshold = -1e-9;
flag_initialize_x0 = true;
rank_init = 1;
rank_max = 12;

C = construct_objective_matrix(F1,F2);

for r= rank_init : rank_max
    % Manopt
    problem.M = quintessentialfactory(r);

    if flag_initialize_x0
        x0 = problem.M.rand();
        flag_initialize_x0 = false;
    else
        x0 = y0;
    end

    problem.cost  = @(x) cost_func(x, C, r);
    problem.egrad = @(x) grad_func(x, C, r);
    problem.ehess = @(x, u) hess_func(x, u, C, r);

    options = [];
    options.verbosity = 0;
    options.miniter = 1; 
    options.maxiter = 300;
    options.maxinner = 500;
    options.tolgradnorm = 5e-5;
    options.rel_func_tol = 5e-7;

    % Solve.
    [x, xcost, ~] = trustregions(problem,x0, options); 
    if verbose
        disp("Rank: "+ num2str(r) + ", Cost: " + num2str(xcost)); 
    end

    % Display some statistics.
    if r == rank_max
        disp("Reached maximum rank! ");
        break
    end
        
    % Compute the Dual Certificate 
    S = find_dual_certificate(x, C, r);

    y = vertcat(x,zeros(4,3));
    
    [v_lm, lambda_lm_original, ~] = eigs(S, 12, 'largestabs');
    lambda_lm = lambda_lm_original;
    lambda_lm = diag(lambda_lm);
    
     % If we are at a saddle point, calculate the escape direction
    if sum(lambda_lm<positive_eigen_value_threshold) > 0
        [lambda_lm, indices] = sort(lambda_lm);
        v_lm = v_lm(:,indices);
        
        indices = lambda_lm < positive_eigen_value_threshold;
        lambda_lm = lambda_lm(indices);
        v_lm = v_lm(:,indices);
        if verbose
            disp("Negative eigenvals: "+mat2str(lambda_lm.'));
            disp("--------------------------------------");
        end
        flag_reduce_success = false;
        problem.M = quintessentialfactory(r+1);
        
        best_xcost = xcost;
        for idx = 1:length(lambda_lm)
            v_curr = v_lm(:, idx);
            y_dot = vertcat(zeros(size(x)), to_stiefel(v_curr,1));
            
            alpha = sqrt(3) * options.tolgradnorm / (norm(v_curr) * abs(lambda_lm(idx)));
            options_ls.ls_max_steps = 20;
            options_ls.ls_contraction_factor = 0.6;
            [~, y0] = linesearch_decrease(problem, y, alpha * y_dot, xcost, [], options_ls);
            current_cost = cost_func(y0,C,r+1);

            if current_cost < best_xcost
                y0_best = y0;
                flag_reduce_success = true;
                best_xcost = current_cost;
            end
        end
        if flag_reduce_success
            y0 = y0_best;
        else
            if verbose
                disp("Failed to escape saddle point!");
            end
            break;
        end
    else
        if verbose
            disp("No negative eigenvalues!");
        end
        lambda_lm = diag(lambda_lm_original);
        indices = lambda_lm < 0;
        lambda_lm = lambda_lm(indices);
        if verbose
            disp(lambda_lm.');
        end
        break
    end
end
if verbose
    disp("Number of Riemannian staircase iterations: " ...
         + num2str(r-rank_init+1));
end


x_bm = to_burer_monteiro(x,r);

X = x_bm*x_bm.';

indices = [1 2 3 5 6 7 9 10 11];
E = round_essential_sdp( X(indices,indices) );
end

function cost = cost_func(V, C, r)
Y = to_burer_monteiro(V, r);
X = Y * Y.';
cost = trace(C.' * X);
end

function grad = grad_func(V, C, r)
grad_f = C;
grad = 2*to_stiefel(grad_f * to_burer_monteiro(V,r),r);
end

function ehess = hess_func(V, U, C, r)
ehess = 2*to_stiefel(C * to_burer_monteiro(U,r),r);
end

function S = find_dual_certificate(V, C, r)
Y = to_burer_monteiro(V,r);
X = Y*Y.';
CX = C * X;

M = zeros(3);

for idx = 1:3
    M(idx,idx) = trace(CX(4*idx-3:4*idx,4*idx-3:4*idx));
    for jdx = idx+1:3
        M(idx,jdx) = trace(CX(4*idx-3:4*idx,4*jdx-3:4*jdx)) + ...
                     trace(CX(4*jdx-3:4*jdx,4*idx-3:4*idx)) ;
    end
end
M_sym = 0.5*(M+M.');
S = C - kron(M_sym,eye(4));
end

function C = construct_objective_matrix(F1,F2)
C = zeros(12);
for i=1:size(F1,2)
    v = kron(F1(:,i),[F2(:,i); 0]);
    C = C + v*v.';
end
end

function V = to_stiefel(Y, r)
V = zeros(4*r,3);
for idx = 1:r
    V(4*idx-3:4*idx, 1) = Y(1:4,idx);
    V(4*idx-3:4*idx, 2) = Y(5:8,idx);
    V(4*idx-3:4*idx, 3) = Y(9:12,idx);
end
end

function Y = to_burer_monteiro(V, r)
Y = zeros(12,r);
for idx = 1:r
    Y(1:4,idx)  = V(4*idx-3:4*idx, 1);
    Y(5:8,idx)  = V(4*idx-3:4*idx, 2);
    Y(9:12,idx) = V(4*idx-3:4*idx, 3);
end
end
