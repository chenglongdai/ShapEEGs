function obj = obj_function(L,L_G,H_cf,S_cf,Gt,A_cf,M_t,W_t,par)
mu_w = reshape(W_t, par.F, 1) .* reshape(M_t, 1, par.channel); 
part1 = 0.5 * trace(L*L_G*L');
part2 = 0.5 * par.gamma1 * sum_HSL(H_cf,S_cf,L,mu_w);
part3 = 0.5 * par.gamma2 * sum(sum(mu_w.*H_cf,[1,2]).^2,'all');
part4 = 0.5 * par.gamma3 * sum_SL_GST(S_cf,L_G,mu_w);
part5 = 0.5 * par.gamma4 * sum(Gt.^2,'all');
part6 = 0.5 * par.gamma5 * sum(sum(mu_w.*A_cf,[1,2]).^2,'all');
obj = part1+part2+part3+part4+part5+part6;
end

function value = sum_HSL(H_cf,S_cf,L,mu_w)
[F,channel,K,V] = size(H_cf);
[~,n] = size(L);
temp = 0;
for f = 1:F
    for c = 1:channel
        H = reshape(H_cf(f,c,:,:),[K,V]);
        S = reshape(S_cf(f,c,:,:),[K,n]);
        temp = temp + mu_w(f,c)*H'*S;
    end
end
value = temp-L;
value = sum(value.^2,'all');
end

function value = sum_SL_GST(S_cf,L_G,mu_w)
[F,channel,K,n] = size(S_cf);
temp = zeros(F,channel);
for f = 1:F
    for c = 1:channel
        S = reshape(S_cf(f,c,:,:),[K,n]);
        temp(f,c) = trace(S*L_G*S');
    end
end
value = sum(mu_w.*temp,[1,2]);
end
