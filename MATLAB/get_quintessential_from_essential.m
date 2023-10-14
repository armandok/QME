function Q = get_quintessential_from_essential(E)
[u,~,v] = svd(E);

Q = zeros(4);
Q(1:3,1:3) = E;
Q(1:3,4) = u(:,3);
Q(4,1:3) = v(:,3);

if det(Q) < 0
    Q(4,1:3) = -v(:,3);
end

end