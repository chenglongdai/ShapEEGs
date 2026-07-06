function [L_G,Gt,G_cf] = compute_G(S_cf,M_t,W_t,par)
n = par.n;
channel = par.channel;
F = par.F;
G_cf = zeros(F,channel,n,n);
mu_w = reshape(W_t, F, 1) .* reshape(M_t, 1, channel); 

for f = 1:F
    for ch = 1:channel
        S = reshape(S_cf(f, ch, :, :),[par.K,n]);
        SS = pdist2(S', S','squaredeuclidean');
        G_cf(f, ch, :, :) = exp(-SS / (2 * par.sigma^2));
    end
end

numL = par.nlabel;
L_label = par.labels(1:numL);
for i=1:numL
    for j=i:numL
        if L_label(i)==L_label(j)
            G_cf(:,:,i,j) = 1;
            G_cf(:,:,j,i) = G_cf(:,:,i,j);
        elseif L_label(i)~=L_label(j)
            G_cf(:,:,i,j) = 0;
            G_cf(:,:,j,i) = G_cf(:,:,i,j);
        end
    end
end


Gt = squeeze(sum(mu_w.*G_cf,[1,2]));

for i=1:numL
    for j=i:numL
        if L_label(i)==L_label(j)
            Gt(i,j)= 1;
            Gt(j,i)= Gt(i,j);
        elseif L_label(i)~=L_label(j)
            Gt(i,j)= 0;
            Gt(j,i)= Gt(i,j);
        end
    end
end
Gt=mapminmax(Gt,0,1);
Gt = tril(Gt) + tril(Gt, -1).';
for i = 1:par.n
    Gt(i,:) = Gt(i,:)/sum(Gt(i,:));
end
% D = diag(sum(Gt, 2));
D = diag(sum(Gt,2)); 

L_G = D - Gt;
end