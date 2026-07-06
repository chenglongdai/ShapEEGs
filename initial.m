function [X_0,M_0,W_0,L_0,H_cf_0,S_cf,S_cf_kit,S_cf_kit_dert] = initial(EEG,data,par)

n = par.n;
channel = par.channel;
F = par.F;
K = par.K;
l = par.l;
q = par.q;
m = par.m;
alpha = par.alpha;
max_l = max(l(:));



X_0 = zeros(F,channel,K,max(l(:)));
for f = 1:F
    for c = 1:channel
        EEG_fc = squeeze(EEG(f,c,:,:));
        [~,EEG_cen] = kmeans(EEG_fc, par.c, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
        Segment_matrix = segment_obtain(EEG_cen, l(f,c));
        [~, X_cf] = kmeans(Segment_matrix, K, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
        temp = zeros(K, max_l);
        temp(:,1:l(f,c)) = X_cf;
        X_0(f,c,:,:) = temp;
    end
end

M_0 = ones(1,channel)/channel;
W_0 = ones(1,F)/F;
%% initial L_0
data0 = zeros(n,channel*m);
for i = 1:n
    for j = 1:channel
        data0(i,1+(j-1)*m : j*m)=data(i,j,:);
    end
end
L_label = par.labels(1:par.nlabel);
data_unlabel=data0(par.nlabel+1:end,:);
[L_unlabel,~] = kmeans(data_unlabel,par.c, 'maxiter', 1000, 'replicates', 20,'emptyaction', 'singleton');

L_ini=[L_label;L_unlabel];
L_0=reshape_L_0(L_ini,par.c);

%% compute S_cf S_cf_kit and its dert
[S_cf,S_cf_kit,S_cf_kit_dert] = compute_S(EEG, X_0, par);



%% initial H_cf
mu_w = reshape(W_0, par.F, 1) .* reshape(M_0, 1, par.channel); 
H_cf_0 = zeros(F,channel,K,par.c);
for f = 1:F
    for c = 1:channel
        H_cf_0(f,c,:,:) = 1/mu_w(f,c)*pinv( reshape( S_cf(f,c,:,:),[K,n] ) )' *L_0';
    end
end

end