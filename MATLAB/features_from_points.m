function [F] = features_from_points(points,rot,position)
F = zeros(size(points));

for idx=1:size(points,2)
    vec = points(:,idx)-position;
    
    F(:,idx) = vec/norm(vec);
end

F = rot.'*F;

end