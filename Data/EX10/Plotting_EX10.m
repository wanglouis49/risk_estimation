clear;clc;

% Poly regr
load('re_poly2_1.mat')
mse_poly2=mse;
bias2_poly2=bias2;
var_poly2=var;
t_poly2=t_tr+t_pr;
load('re_poly5_1.mat')
mse_poly5=mse;
bias2_poly5=bias2;
var_poly5=var;
t_poly5=t_tr+t_pr;
load('re_poly8_1.mat')
mse_poly8=mse;
bias2_poly8=bias2;
var_poly8=var;
t_poly8=t_tr+t_pr;
load('re_poly10_1.mat')
mse_poly10=mse;
bias2_poly10=bias2;
var_poly10=var;
t_poly10=t_tr+t_pr;
load('re_poly15_1.mat')
mse_poly15=mse;
bias2_poly15=bias2;
var_poly15=var;
t_poly15=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_poly2,K,mse_poly5,K,mse_poly8,K,mse_poly10,K,mse_poly15)
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','deg=10','deg=15')

figure(2),loglog(K,bias2_poly2,K,bias2_poly5,K,bias2_poly8,K,bias2_poly10,K,bias2_poly15)
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','deg=10','deg=15')

figure(3),loglog(K,var_poly2,K,var_poly5,K,var_poly8,K,var_poly10,K,var_poly15)
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','deg=10','deg=15')

figure(4),loglog(K,t_poly2,K,t_poly5,K,t_poly8,K,t_poly10,K,t_poly15)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('deg=2','deg=5','deg=8','deg=10','deg=15')

%% Ridge Regr
clear; clc
load('re_poly2_100_qmc.mat')
mse_poly=mse;
bias2_poly=bias2;
var_poly=var;
t_poly=t_tr+t_pr;
load('re_ridge2_100_qmc.mat')
mse_ridge=mse;
bias2_ridge=bias2;
var_ridge=var;
t_ridge=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K(3:end),mse_poly(3:end),K(3:end),mse_ridge(3:end),...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('Poly Regr','Ridge Regr','k^{-1}')

figure(2),loglog(K(3:end),bias2_poly(3:end),K(3:end),bias2_ridge(3:end),...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1e3])
legend('Poly Regr','Ridge Regr','k^{-1}')

figure(3),loglog(K(3:end),var_poly(3:end),K(3:end),var_ridge(3:end),...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1e3])
legend('Poly Regr','Ridge Regr','k^{-1}')

figure(4),loglog(K,t_poly,K,t_ridge)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('Poly Regr','Ridge Regr')

%% N_o
load('re_poly15_1_qmc.mat')
mse_1=mse;
bias2_1=bias2;
var_1=var;
t_1=t_tr+t_pr;
load('re_poly15_10_qmc.mat')
mse_10=mse;
bias2_10=bias2;
var_10=var;
t_10=t_tr+t_pr;
load('re_poly15_100_qmc.mat')
mse_100=mse;
bias2_100=bias2;
var_100=var;
t_100=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_1,K(2:end),mse_10(2:end),K(5:end),mse_100(5:end),...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(2),loglog(K,bias2_1,K,bias2_10,K,bias2_100,...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1e3])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(3),loglog(K,var_1,K,var_10,K,var_100,...
K_plot,5e-1*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1e3])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(4),loglog(K,t_1,K,t_10,K,t_100)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('N_i=1','N_i=10','N_i=100')