function X_t = Update_shapelet(EEG,X_t,L,L_G,H_cf,S_cf,S_cf_kit,S_cf_kit_dert,Gt,A_cf,A_dert,M,W,par)
    F = par.F;
    channel = par.channel;
    K = par.K;
    n = par.n;
    q = par.q;
    l = par.l;

    mu_w = reshape(W, F, 1) .* reshape(M, 1, channel);

    Part1 = zeros(F,channel,K,max(l(:)));
    Part2 = zeros(F,channel,K,max(l(:)));
    Part3 = zeros(F,channel,K,max(l(:)));
    Part4 = zeros(F,channel,K,max(l(:)));
    Part5 = zeros(F,channel,K,max(l(:)));
    Part6 = zeros(F,channel,K,max(l(:)));
    parameter1=1/2*L'*L;
    parameter5=par.gamma4*Gt;
    tic;
    for f = 1:F
        for c = 1:channel
            S = squeeze(S_cf(f,c,:,:));
            parameter3=0.5*par.gamma3* S'*S;
            parameter6=par.gamma5*squeeze(A_cf(f,c,:,:));
    
            for k = 1:K
                for h = 1:l(f,c)
                    G_dert = zeros(n,n,K,l(f,c));
                    for i = 1:n
                        for j = 1:n
                            G_dert(i,j,k,l) = Gt(i,j)*(-1/par.sigma^2)*(S_cf(f,c,k,i)-S_cf(f,c,k,j))...
                                *(S_cf_kit_dert(f,c,k,i,h)-S_cf_kit_dert(f,c,k,j,h));
                            % if j==i
                            %     P1(i,j) = parameter1(i,j)*sum(G_dert(i,:,k,h));
                            %     P3(i,j) = M(c)*W(f)*parameter3(i,j) * sum(G_dert(i,:,k,h));
                            %     P5(i,j) = parameter5(i,j) * sum(G_dert(i,:,k,h));
                            % else
                                P1(i,j)= parameter1(i,j)*(-1*G_dert(i,j,k,h));
                                P3(i,j) = M(c)*W(f)*parameter3(i,j) * (-1*G_dert(i,j,k,h));
                                P5(i,j) = parameter5(i,j) * G_dert(i,j,k,h);
                            % end
                        end
                    end
                    Part1(f,c,k,h)=sum(sum(P1));
                    Part3(f,c,k,h)=sum(sum(P3));
                    Part6(f,c,k,h)=2*M(c)*W(f)*sum(parameter6(k,:).*squeeze(A_dert(f,c,k,:,h))')-parameter6(k,k)*A_dert(f,c,k,k,h);
                end
            end
        end
    end
    toc
    while t <=par.max_iter
          X_t = X_t - par.learn_rate * X_dert;
          t = t+1;
    end
end

