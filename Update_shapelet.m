function X_t = Update_shapelet( X_t, L, L_G, H_cf, S_cf,  S_cf_kit_dert, Gt,G_cf, A_cf, A_dert, M, W, par)
    F = par.F;
    channel = par.channel;
    K = par.K;
    n = par.n;
    l_mat = par.l;
    V = par.c;
    sigma = par.sigma;
    max_l = max(l_mat(:));
    
    mu_w = reshape(W, F, 1) .* reshape(M, 1, channel);
    
    mu_w_exp = repmat(mu_w, [1, 1, K, K]); 
    sum_A = reshape(sum(sum(A_cf .* mu_w_exp, 1), 2), [K, K]);
    

    HSL = zeros(V, n);
    for f = 1:F
        for c = 1:channel
            H_sub = reshape(H_cf(f,c,:,:), [K, V]);
            S_sub = reshape(S_cf(f,c,:,:), [K, n]);
            HSL = HSL + mu_w(f,c) * (H_sub' * S_sub);
        end
    end
    HSL = HSL - L;
    
    parameter1 = 0.5 * (L' * L);
    parameter5 = par.gamma4 * Gt;
    
    Total_Part = zeros(F, channel, K, max_l); 

    parfor f = 1:F
        local_grad_f = zeros(1, channel, K, max_l);
        
        for c = 1:channel
            lf_c = l_mat(f, c);
            mu_val = mu_w(f, c);

            S = reshape(S_cf(f, c, :, :), [K, n]); 
            H = reshape(H_cf(f, c, :, :), [K, V]);
            G = reshape(G_cf(f, c, :, :), [n, n]);
            
            S_der_tensor = reshape(S_cf_kit_dert(f, c, :, :, 1:lf_c), [K, n, lf_c]);
            A_d_tensor = reshape(A_dert(f, c, :, :, 1:lf_c), [K, K, lf_c]);
            
            % Part 2 & 4 
            term2_base = (par.gamma1 * mu_val) * HSL;     % [V, n]

            term4_base = V * (par.gamma3 * mu_val) * (S * L_G); % [K, n]
            
            % Part 3 
            param3 = 0.5 * par.gamma3 * mu_val * (S' * S);
            P_total = parameter1 + param3 + parameter5;

            G_base = (mu_val * (-1 / sigma^2)) * G;
            
            for k = 1:K
                S_k = S(k, :); 

                % S_diff = Si - Sj
                S_diff = bsxfun(@minus, S_k', S_k); 
                
                % M = P_total .* G_base .* S_diff
                M_mat = P_total .* G_base .* S_diff;

                if par.nlabel > 0
                    M_mat(1:par.nlabel, 1:par.nlabel) = 0;
                end

                % sum(M .* (Di - Dj)) = D' * (RowSum - ColSum)
                % RowSum[i] = sum(M(i, :))
                % ColSum[j] = sum(M(:, j))
                RowSum = sum(M_mat, 2);    % [n, 1]
                ColSum = sum(M_mat, 1).';  % [n, 1]
                v_reduction = (RowSum - ColSum).'; % [1, n]
                
                vec_24 = H(k, :) * term2_base + term4_base(k, :);
                p6_val = par.gamma5 * mu_val * sum_A(k, :);
                D_all_h = reshape(S_der_tensor(k, :, :), [n, lf_c]);
                A_d_slice = reshape(A_d_tensor(k, :, :), [K, lf_c]);
                res_135 = v_reduction * D_all_h;
                res_24 = vec_24 * D_all_h;
                res_6 = p6_val * A_d_slice;
                local_grad_f(1, c, k, 1:lf_c) = res_135 + res_24 + res_6;
            end
        end
        Total_Part(f, :, :, :) = local_grad_f;
    end
    

    X_dert = Total_Part;
    t = 1;
    while t <= par.max_iter
        X_t = X_t - par.learn_rate * X_dert;
        t = t + 1;
    end
end
