function [E_estimate] = round_essential_sdp(X)

[U, ~, ~] = svd(X);
u = U(:,1);

E_estimate = zeros(3);
E_estimate(1,:) = u(1:3);
E_estimate(2,:) = u(4:6);
E_estimate(3,:) = u(7:9);
E_estimate = project_essential(E_estimate);
end

function [E,v1,v2] = project_essential(EE)
[u,s,v] = svd(EE);
s(1,1)=1;
s(2,2)=1;
s(3,3)=0;

E = u*s*v.';
v1 = null(E);
v2 = null(E.');
end