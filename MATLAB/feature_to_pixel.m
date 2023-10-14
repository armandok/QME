function [pixels] = feature_to_pixel(F, f)
num = size(F,2);
pixels = zeros(2,num);
for i=1:num
    pixels(:,i) = F(1:2,i)*f/F(3,i);
end

end