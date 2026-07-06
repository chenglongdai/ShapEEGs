function [X_opt,L_opt,M_opt,W_opt,Obj] = ShapEEGs(EEG,data,par)
EEG = z_regularization(EEG);
[X_0,M_0,W_0,L_0,H_cf_0,S_cf_0,S_cf_kit,S_cf_kit_dert]  = initial(EEG,data,par);

[A_cf_0,A_dert] = compute_A(X_0,par);
[L_G,Gt,G_cf] = compute_G(S_cf_0,M_0,W_0,par);
X_t = X_0;
M_t = M_0;
W_t = W_0;
L_t = L_0;
H_cf_t = H_cf_0;
S_cf_t = S_cf_0;
S_cf_kit_t = S_cf_kit;
A_cf_t = A_cf_0;
iter = 0;
while true
    iter = iter+1;

    Obj(iter) = obj_function(L_t,L_G,H_cf_t,S_cf_t,Gt,A_cf_t,M_t,W_t,par);
    if iter > 2 &&   abs((Obj(iter) - Obj(iter-1) )/Obj(iter-1))< par.epsilon 
        break;
    end

    X_t = Update_shapelet(X_t,L_t,L_G,H_cf_t,S_cf_t,S_cf_kit_dert,Gt,G_cf,A_cf_t,A_dert,M_t,W_t,par);
    X_t = z_regularization(X_t);
    [M_t,W_t] = Update_MW(M_t,W_t,S_cf_kit_t,par); 
    [S_cf_t,S_cf_kit_t,S_cf_kit_dert] = compute_S(EEG,X_t,par);
    H_cf_t = Update_H(H_cf_t,S_cf_t,L_t,M_t,W_t,par);
    [L_G,Gt,G_cf] = compute_G(S_cf_t,M_t,W_t,par);
    L_t = Update_L(H_cf_t,S_cf_t,L_G,M_t,W_t,L_t,par);
    [A_cf_t,A_dert] = compute_A(X_t,par);

    if iter>par.max_iter
        break;
    end

end

[mL,nL]=size(L_t);

for i=1:nL
    L_max=max(L_t(:,i));
    for j=1:mL
        if L_t(j,i)==L_max
            L_t(j,i)=1;
        else
            L_t(j,i)=0;
        end
    end
end
X_opt = X_t;
L_opt = L_t;
M_opt = M_t';
W_opt = W_t';
end