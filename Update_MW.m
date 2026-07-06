function [M_t,W_t] = Update_MW(M,W,S_cf_kit,par)
    channel = par.channel;
    F = par.F;
    K = par.K;
    n = par.n;

    b_logits = zeros(channel, K); 
    p_logits = zeros(F, K);

    M_curr = M;
    W_curr = W;
    
    ITER_NUM = 3; 
    
    for iter = 1:ITER_NUM

        % M: (1, C, 1, 1, 1)
        mu_expand = reshape(M_curr, 1, channel, 1, 1, 1); 
        % W: (F, 1, 1, 1, 1)
        w_expand = reshape(W_curr, F, 1, 1, 1, 1);
        
        mu_w_full = mu_expand .* w_expand;
        
        S_weighted = mu_w_full .* S_cf_kit;
        
        S_sum_FC = sum(S_weighted, [1, 2]);
        
        min_S = min(S_sum_FC, [], 5); 
        % exp( alpha * (D - D_min) )
        exp_term = exp(par.alpha * (S_sum_FC - min_S)); 
        
        denominator = sum(exp_term, 5); 
        denominator = denominator + eps; 
        
        % sum_t ( (sum_f w^f * S) * exp(...) )
        % sum_f w^f * S -> (1, C, K, N, q)
        term_C = sum(w_expand .* S_cf_kit, 1); 
        numerator_C = sum(term_C .* exp_term, 5); % (1, C, K, N)
        S_cf_C = numerator_C ./ denominator; % (1, C, K, N)
        S_cf_C = reshape(S_cf_C, [channel, K, n]); 
        
        % sum_t ( (sum_c mu^c * S) * exp(...) )
        % sum_c mu^c * S -> (F, 1, K, N, q)
        term_F = sum(mu_expand .* S_cf_kit, 2);
        numerator_F = sum(term_F .* exp_term, 5); % (F, 1, K, N)
        S_cf_F = numerator_F ./ denominator; % (F, 1, K, N)
        S_cf_F = reshape(S_cf_F, [F, K, n]);
        
        numerator_S = sum(S_sum_FC .* exp_term, 5);
        S_global = numerator_S ./ denominator; % (1, 1, K, N)
        S_global = reshape(S_global, [K, n]);
        
        % S_global: (K, n)
        row_norms = sqrt(sum(S_global.^2, 2)); % (K, 1)
        safe_norms = row_norms + eps; 
        scale = (row_norms.^2) ./ (1 + row_norms.^2) ./ safe_norms;
        S_hat = S_global .* scale; % (K, n)
        
        % b_k^c += S_hat_k * (S_cf_C_k)^T
        % S_hat: (K, n), S_cf_C: (C, K, n)
        
        for c = 1:channel
            % S_cf_C(c,:,:) -> (K, n)
            S_c_slice = squeeze(S_cf_C(c, :, :)); 
            dot_prod = sum(S_hat .* S_c_slice, 2); % (K, 1)
            b_logits(c, :) = b_logits(c, :) + dot_prod'; 
        end
        
        for f = 1:F
            S_f_slice = squeeze(S_cf_F(f, :, :));
            dot_prod = sum(S_hat .* S_f_slice, 2);
            p_logits(f, :) = p_logits(f, :) + dot_prod';
        end
        
        b_norms = sqrt(sum(b_logits.^2, 2)); % (C, 1)
        p_norms = sqrt(sum(p_logits.^2, 2)); % (F, 1)
        
        % M = gamma/C + (1-gamma) * Softmax(b_norms)
        
        b_norms_safe = b_norms - max(b_norms);
        exp_b = exp(b_norms_safe);
        softmax_b = exp_b / sum(exp_b);
        M_curr = par.r / channel + (1 - par.r) * softmax_b;
        M_curr = M_curr / sum(M_curr);
        
        p_norms_safe = p_norms - max(p_norms);
        exp_p = exp(p_norms_safe);
        softmax_p = exp_p / sum(exp_p);
        W_curr = par.p / F + (1 - par.p) * softmax_p;
        W_curr = W_curr / sum(W_curr);
    end
    
    M_t = M_curr;
    W_t = W_curr;
end


