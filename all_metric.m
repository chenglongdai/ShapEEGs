function [Result] = all_metric(L_opt,L_true,par)

NMI_1=[];
NMI_2=[];
SR_t=[];
SAR_t=[];
microfscore=[];
macrofscore=[];
ka = [];
for i=1:par.c
    for j=1:par.c
        NMI_1=[NMI_1 NMI(L_opt(i,:),L_true(j,:))];
        [zRand,SR,SAR,VI]=zrand(L_opt(i,:),L_true(j,:));
        SR_t=[SR_t SR];
        SAR_t=[SAR_t SAR];
        [micro, macro]=micro_macro_PR(L_opt(i,:),L_true(j,:));
        microfscore=[microfscore micro.fscore];
        macrofscore=[macrofscore macro.fscore];
        kappa= kappaindex(L_opt(i,:)+1,L_true(j,:)+1,par.c);
        ka=[ka kappa];
    end
end
NMI_1=max(NMI_1);
SAR_F=max(SAR_t);
microfscore=max(microfscore);
macrofscore=max(macrofscore);
kappa= max(ka);
RI = RandIndex(L_opt,L_true); % Calculate Rand Index
Result=[RI NMI_1 SAR_F microfscore macrofscore kappa];
end