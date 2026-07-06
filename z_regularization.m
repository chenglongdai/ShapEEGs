function Data_z=z_regularization(Data)

[f,c,n,m] = size(Data);
Data_z = zeros(f,c,n,m);

for i = 1:f
    for j = 1:c
        Data_z(i,j,:,:) = mapminmax(reshape(Data(i,j,:,:),[n,m]),0,1);
    end
end
end
