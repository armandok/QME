function [e] = error_essential_algebraic(E,F1,F2)
e = 0;
for idx = 1:size(F1,2)
    e = e + (F1(:,idx).' * E * F2(:,idx))^2;
end
end