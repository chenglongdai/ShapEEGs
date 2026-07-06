clear all;
close all;
clc;
warning off;
addpath(genpath('./'));
addpath('performance\');
path = 'dataset/';
dataset = 'Data_II_Ia';
load ([path dataset,'.mat']);

data=eval(dataset);
L_true = labels;

[par.n,par.channel,par.m] = size(data);

par.sampling_rate = sampling_rate;      % Sampling rate
par.c = max(L_true(:));                    % clustering number
par.F = 5;
par.labels = labels;
par.ratio = 0; %0% 5% 10% 20% 30% 40% 50%
par.nlabel=floor(par.ratio*par.n); % random setting for number of labeled EEG signal

par.K = 2;                    % number of shapelet

par.l = par.sampling_rate*ones(par.F,par.channel)/2;
par.q = par.m-par.l+1;

par.r = 0.5;                    % trade-off parameter
par.p = 0.5;                % trade-off parameter
par.alpha = -100; %100; % -60;                 % distance parameter
par.sigma = 1; % 0.01;                 % Gaussan kernel

par.gamma1 = 1;               % regularization parameters
par.gamma2 = 1;
par.gamma3 = 1;
par.gamma4 = 1;
par.gamma5 = 1;

par.learn_rate = 0.001;       % learn rate
par.max_iter = 50;            % maximum itertions 
par.epsilon = 0.01;

L_true=L_true(par.nlabel+1:end,:);
L_true = l_true_reshape(L_true,par.c);



count = 5;
Result=zeros(6,count);

for o = 1:count
    tic;
    [X_opt,L_opt,M_opt,W_opt,Obj] = ShapEEGs(EEG,data,par);
    L_opt = L_opt(:,par.nlabel+1:end);
    Result(:,o) = all_metric(L_opt,L_true,par);
end

