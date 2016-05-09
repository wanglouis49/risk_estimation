clear;clc;

% Poly Regr
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
load('re_spec_full_1.mat')
mse_spec=mse;
bias2_spec=bias2;
var_spec=var;
t_spec=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_poly2,K,mse_poly5,K,mse_poly8,K,mse_spec,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1])
legend('deg=2','deg=5','deg=8','spec')

figure(2),loglog(K,bias2_poly2,K,bias2_poly5,K,bias2_poly8,K,bias2_spec,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1])
legend('deg=2','deg=5','deg=8','spec')

figure(3),loglog(K,var_poly2,K,var_poly5,K,var_poly8,K,var_spec,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1])
legend('deg=2','deg=5','deg=8','spec')

figure(4),loglog(K,t_poly2,K,t_poly5,K,t_poly8,K,t_spec)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('deg=2','deg=5','deg=8','spec')

%% N_o
load('re_poly8_1.mat')
mse_1=mse;
bias2_1=bias2;
var_1=var;
t_1=t_tr+t_pr;
load('re_poly8_10.mat')
mse_10=mse;
bias2_10=bias2;
var_10=var;
t_10=t_tr+t_pr;
load('re_poly8_100.mat')
mse_100=mse;
bias2_100=bias2;
var_100=var;
t_100=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_1,K,mse_10,K,mse_100,...
K_plot,1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(2),loglog(K,bias2_1,K,bias2_10,K,bias2_100,...
K_plot,1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(3),loglog(K,var_1,K,var_10,K,var_100,...
K_plot,1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1])
legend('N_i=1','N_i=10','N_i=100','k^{-1}')

figure(4),loglog(K,t_1,K,t_10,K,t_100)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('N_i=1','N_i=10','N_i=100')