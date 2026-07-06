function [A_cf,A_dert]  = compute_A(X,par)
n = par.n;
channel = par.channel;
F = par.F;
K = par.K;
l = par.l;
q = par.q;
m = par.m;
alpha = par.alpha;


A_cf = zeros(F,channel,K,K);
A_dert = zeros(F,channel,K,K,max(l(:)));

for f = 1:F
    for c = 1:channel
        diff_shapelet = zeros(K,K,max(l(:)));
        dist = zeros(K,K);
        for i = 1:K
            for j = 1:K
                diff = reshape(X(f,c,i,1:l(f,c))-X(f,c,j,1:l(f,c)),[1,l(f,c)]);
                diff_shapelet(i,j,1:l(f,c)) = -2*diff;
                tmp =diff .^2;  
                tmp = sum(tmp)/par.l(f,c);
                dist(i,j) = tmp;
                A_cf(f,c,i,j) = exp(-(tmp.^2/(2*par.sigma^2)));
            end
        end
        Q1=exp(alpha*dist);
        Q2=dist .* Q1;
        temp1 = zeros(K,K,l(f,c));
        for i = 1:K
            for j = 1:K
                part1 = 1/(Q1(i,j)^2);
                part2 = exp(alpha*dist(i,j))*((1+alpha*dist(i,j))*Q1(i,j) - alpha*Q2(i,j));
                A_tmp1 = exp(-dist(i,j)^2/(2*par.sigma^2));
                temp1(i,j,:) = part1*part2.*diff_shapelet(i,j,1:l(f,c));
                A_dert(f,c,i,j,1:l(f,c)) = (-dist(i,j)/par.sigma^2)*A_tmp1*temp1(i,j,1:l(f,c));
            end
        end
    end
end

end