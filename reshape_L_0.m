function L_0_matrix=reshape_L_0(L_0,c)
index=find(L_0==0);
L_0(index)=c;
index1=find(L_0==-1);
L_0(index1)=c;

mL_0=length(L_0);
L_0_matrix=zeros(c,mL_0);

for i=1:mL_0
    L_0_matrix(L_0(i),i)=1;
end