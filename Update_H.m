function H_cf = Update_H(H_cf, S_cf, L, M_t, W_t, par)
    F = par.F;
    channel = par.channel;
    K = par.K;
    c_dim = par.c; 
    n = par.n;

    batch_size = F * channel;
    
    mu_w = reshape(W_t, F, 1) .* reshape(M_t, 1, channel); 
    mu_flat = reshape(mu_w, [1, 1, batch_size]);
    
    H_flat = permute(H_cf, [3, 4, 1, 2]);
    H_flat = reshape(H_flat, K, c_dim, batch_size);
    
    S_flat = permute(S_cf, [3, 4, 1, 2]);
    S_flat = reshape(S_flat, K, n, batch_size);
    
    gamma1 = par.gamma1;
    gamma2 = par.gamma2;
    lr = par.learn_rate;
    
    for iter = 1:par.max_iter
        H_weighted = bsxfun(@times, H_flat, mu_flat);
        
        HTS_batch = pagemtimes(H_weighted, 'transpose', S_flat, 'none');

        temp_sum = sum(HTS_batch, 3);
        

        HSL = temp_sum - L; % [V, n]
        

        sum_H = sum(H_weighted, 3);
        Term1 = pagemtimes(S_flat, 'none', HSL', 'none');
        Term1 = bsxfun(@times, Term1, gamma1 * mu_flat);
        Term2 = bsxfun(@times, sum_H, gamma2 * mu_flat);
        H_dert = Term1 + Term2;
        H_flat = H_flat - lr * H_dert;
    end
    
    H_cf = reshape(H_flat, K, c_dim, F, channel);
    H_cf = permute(H_cf, [3, 4, 1, 2]);
end