function [E, X] = solve_essential_local(F1, F2, x0)

N_pp = size(F1,2);
F = zeros(3,3,N_pp);
for i=1:N_pp
    F(:,:,i) = F1(:,i)*F2(:,i).';
end

% set the manifold for Manopt
manifold = quintessentialqo2factory();
problem.M = manifold;

if isempty(x0)
    x0 = manifold.rand();
else
    assert(all(size(x0) == [4 4 2]));
end

problem.cost  = @(x) cost_func(x, F);
problem.egrad = @(x) grad_func(x, F);
problem.ehess = @(x, u) hess_func(x, u, F);


options = [];
options.verbosity = 0;
options.maxiter = 200; 
options.tolgradnorm = 1e-4;
options.rel_func_tol = 1e-5;

[x, ~, ~] = trustregions(problem,x0, options);

Q = x(:,:,1).'*x(:,:,2);
E = Q(1:3,1:3);

X = Q;
end


function cost = cost_func(x, F)

cost = 0;
E = x(:,1:3,1).'*x(:,1:3,2);

for i=1:size(F,3)
    cost = cost + sum(F(:,:,i).*E, 'all')^2;
end
end

function grad = grad_func(y, F)
grad_f = grad_of_f(y, F);
grad = zeros(size(y));
grad(:,:,1) = 2*y(:,:,2) * grad_f.';
grad(:,:,2) = 2*y(:,:,1) * grad_f;
end

function hess = hess_func(y, y_dot, F)
grad_f = grad_of_f(y, F);
grad = zeros(size(y));
grad(:,:,1) = 2*y(:,:,2) * grad_f.';
grad(:,:,2) = 2*y(:,:,1) * grad_f;

hess = zeros(size(y));
hess_f = hess_of_f(y, y_dot, F); % ???

hess(:,:,1) = 2*y(:,:,2) *hess_f(:,:,1).';
hess(:,:,2) = 2*y(:,:,1) *hess_f(:,:,1);
end

function g_f = grad_of_f(y, F)

g_f = zeros(4);

E = y(:,1:3,1).'*y(:,1:3,2);

C = zeros(3);
for i=1:size(F,3)
    C = C + sum(F(:,:,i).*E, 'all')*F(:,:,i);
end

g_f(1:3,1:3) = C;
end

function h_f = hess_of_f(y, y_dot, F)

h_f = zeros(size(y));

Q_dot = zeros(size(y));

Q_dot(:,:,1) = y_dot(:,:,1).'*y(:,:,2);
Q_dot(:,:,2) = y(:,:,1).'*y_dot(:,:,2);

C1 = zeros(3);
C2 = zeros(3);
for i=1:size(F,3)
    C1 = C1 + sum(F(:,:,i).*Q_dot(1:3,1:3,1), 'all')*F(:,:,i);
    C2 = C2 + sum(F(:,:,i).*Q_dot(1:3,1:3,2), 'all')*F(:,:,i);
end

h_f(1:3,1:3,1) = C1;
h_f(1:3,1:3,2) = C2;
end