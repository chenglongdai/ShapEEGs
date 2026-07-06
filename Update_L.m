function L_t = Update_L(H_cf,S_cf,L_G,M_t,W_t,L_0,par)
L_t = L_0;
mu_w = reshape(W_t, par.F, 1) .* reshape(M_t, 1, par.channel); 
tmp2 = zeros(par.c, par.n);
for f = 1:par.F
    for c = 1:par.channel
        H = reshape(H_cf(f, c, :, :),[par.K,par.c]);  % k x v
        S = reshape(S_cf(f, c, :, :),[par.K,par.n]);  % k x n
        tmp2 = tmp2 + mu_w(f,c) * (H') * S;  % v x n
    end
end
for c = 1:par.c
    tmp2(c,:) = tmp2(c,:)/sum(tmp2(c,:));
end
t=1;
while t<par.max_iter+1
    P1 = L_t*L_G;
    P2 = par.gamma1*(tmp2-L_t);
    dertL = P1-P2;
    L_t = L_t-par.learn_rate*dertL;
    t = t+1;
end
end


