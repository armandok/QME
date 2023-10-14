function [X_e, X_t] = solve_essential_sdp(F1, F2)
% assert(size(F1) == size(F2));
N = size(F1, 2);

X = sdpvar(12,12,'symmetric');


Constraints = X >= zeros(12);

for i = 0:2
    Constraints = [Constraints, ...
        X(i*4+1,i*4+1)+X(i*4+2,i*4+2)+X(i*4+3,i*4+3)+X(i*4+4,i*4+4)==1];
end
Constraints = [Constraints, X(4,4)+X(8,8)+X(12,12)==1];

% Orthogonality constraints
Constraints = [Constraints, X(1,5)+X(2,6)+X(3,7)+X(4,8)==0];
Constraints = [Constraints, X(1,9)+X(2,10)+X(3,11)+X(4,12)==0];
Constraints = [Constraints, X(5,9)+X(6,10)+X(7,11)+X(8,12)==0];

C = zeros(12);
for i=1:N
    v = kron( F1(:,i) , [F2(:,i);0] );
    C = C + v*v.';
end

options = sdpsettings('verbose',0);
diagnostics = optimize(Constraints, trace(C.' * X),options);

% Get the optimized semidefinite solution
sol = value(X);

% Extract the two diagonal blocks from sol into X_e and
% X_t. Other entries in sol should be zero, due to the degeneracy in the
% SDP problem.
indices = [1 2 3 5 6 7 9 10 11];
X_e = sol(indices, indices);
X_t = sol(4:4:12,4:4:12);
end