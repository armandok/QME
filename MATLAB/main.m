clc
clear
close all

% Camera parameters
k = [800 0 0; 0 800 0; 0 0 1];
f = 800;

% 3D point gen params
scale = 4;
transport = [0;0;0.5]*scale;

% Camera positions
T1 = randn(3,1);
T2 = randn(3,1);
t12 = (T2-T1)/norm(T2-T1);
tx = [0, -t12(3), t12(2); t12(3), 0, -t12(1); -t12(2), t12(1), 0];

% Number of correspondence
N = 15;

% Noise level in pixels
sigma = 5;

% Sample a random rotation for the second camera's rotation
R1 = eye(3);
[R2,~] = qr(randn(3));
R2 = R2*diag([1,1,det(R2)]); % Ensure that R2 is a rotation

% Sample 3D points
P = (rand(3,N)-0.5*[1;1;1])*scale+transport;

% Essential matrix, ground truth
E_gt = R1.'*tx*R2;

% Features (bearings)
F1 = features_from_points(P,R1,T1);
F2 = features_from_points(P,R2,T2);

% Pixel coordinates of the feature points
PX1 = feature_to_pixel(F1, f);
PX2 = feature_to_pixel(F2, f);

% Add noise to pixel coordinates
for i=1:N
    PX1(:,i) = PX1(:,i) + randn(2,1)*sigma; 
    PX2(:,i) = PX2(:,i) + randn(2,1)*sigma;
    F1(:,i) = [PX1(:,i); f];
    F1(:,i) = F1(:,i)/norm(F1(:,i));
    F2(:,i) = [PX2(:,i); f];
    F2(:,i) = F2(:,i)/norm(F2(:,i));
end

% Solve using the Riemannian staircase method
tic
[E_est, X] = solve_essential_staircase(F1, F2, 'verbose', false);
t_bm = toc;

% Take the solution and do local optimization to compensate for rounding
% error
y0 = zeros(4,4,2);
y0(:,:,1) = eye(4);

y0(:,:,2) =get_quintessential_from_essential(E_est);
tic
[E_lcl, X_lcl] = solve_essential_local(F1, F2, y0);
t_lcl = toc;

% Solve using an SDP solver. This step uses YALMIP.
tic;
[X_sdp_e, X_sdp_t] = solve_essential_sdp(F1, F2);
E_sdp = round_essential_sdp(X_sdp_e);
t_sdp = toc;


R12 = R1.'*R2;
t_gt = R1.'*t12;

% Display runtimes
disp("SDP runtime:             " + num2str(t_sdp));
disp("Staircase runtime:       " + num2str(t_bm));
disp("Staircase+local runtime: " + num2str(t_bm+t_lcl));

disp(" ");

disp("SDP algebraic error: " + ...
    num2str(error_essential_algebraic(E_sdp,F1,F2)));
disp("Staircase algebraic error: " + ...
    num2str(error_essential_algebraic(E_est,F1,F2)));
disp("Staircase algebraic error: " + ...
    num2str(error_essential_algebraic(E_lcl,F1,F2)));

