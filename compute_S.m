function [S_cf, S_cf_kit, S_cf_kit_dert] = compute_S(EEG, X_t, par) 
    n = par.n;             
    channel = par.channel; 
    F = par.F;             
    K = par.K;             
    l = par.l;             
    q = par.q;             
    alpha = par.alpha;

    max_q = max(q(:));
    max_l = max(l(:));
    
    S_cf_kit = zeros(F, channel, K, n, max_q);
    S_cf_kit_dert = zeros(F, channel, K, n, max_l);

    parfor f = 1:F
        local_S_cf_kit = zeros(1, channel, K, n, max_q);
        local_S_cf_kit_dert = zeros(1, channel, K, n, max_l);
        
        for c = 1:channel
            q_fc = q(f,c); 
            l_fc = l(f,c); 
            
            EEG_data = squeeze(EEG(f,c,:,:)); 
            Shapelets = squeeze(X_t(f, c, :, 1:l_fc));
            if K == 1, Shapelets = Shapelets'; end

            S_sq = sum(Shapelets.^2, 2); 
            ones_kernel = ones(1, l_fc);
            E_sq_full = conv2(EEG_data.^2, ones_kernel, 'valid');
            E_sq = E_sq_full(:, 1:q_fc);
            
            dist_sq = zeros(K, n, q_fc);
            for k = 1:K
                S_k = Shapelets(k, :);
                inter = conv2(EEG_data, fliplr(S_k), 'valid');
                inter = inter(:, 1:q_fc);
                d_val = bsxfun(@plus, E_sq, S_sq(k)) - 2 * inter;
                dist_sq(k, :, :) = d_val / l_fc;
            end
            
            local_S_cf_kit(1, c, :, :, 1:q_fc) = permute(dist_sq, [4, 5, 1, 2, 3]);

            temp0 = dist_sq;
            exp_A = exp(alpha * temp0);
            Q1 = sum(exp_A, 3);
            Q2 = sum(temp0 .* exp_A, 3);
            Part1 = 1 ./ (Q1.^2); % K x n
            
            Q1_exp = repmat(Q1, [1, 1, q_fc]);
            Q2_exp = repmat(Q2, [1, 1, q_fc]);
            W_all = exp_A .* (Q1_exp .* (1 + alpha * temp0) - alpha * Q2_exp); % K x n x q
            Sum_W = sum(W_all, 3); % K x n
            
            %  C++ 
            local_grad = compute_S_mexfunction(EEG_data, Shapelets, W_all, Part1, Sum_W);
          
            local_S_cf_kit_dert(1, c, :, :, 1:l_fc) = reshape(local_grad, [1, 1, K, n, l_fc]);
        end
        
        S_cf_kit(f, :, :, :, :) = local_S_cf_kit;
        S_cf_kit_dert(f, :, :, :, :) = local_S_cf_kit_dert;
    end

    exp_term = exp(par.alpha * S_cf_kit);          
    numerator = sum(S_cf_kit .* exp_term, 5);                 
    denominator = sum(exp_term, 5);                           
    S_cf = numerator ./ denominator;                         
end
