clear;clc;

% N_o = 1
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

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_poly2,K,mse_poly5,K,mse_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(2),loglog(K,bias2_poly2,K,bias2_poly5,K,bias2_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(3),loglog(K,var_poly2,K,var_poly5,K,var_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(4),loglog(K,t_poly2,K,t_poly5,K,t_poly8)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('deg=2','deg=5','deg=8','spec')

%% N_o = 10
clear; clc
load('re_poly2_10.mat')
mse_poly2=mse;
bias2_poly2=bias2;
var_poly2=var;
t_poly2=t_tr+t_pr;
load('re_poly5_10.mat')
mse_poly5=mse;
bias2_poly5=bias2;
var_poly5=var;
t_poly5=t_tr+t_pr;
load('re_poly8_10.mat')
mse_poly8=mse;
bias2_poly8=bias2;
var_poly8=var;
t_poly8=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_poly2,K,mse_poly5,K,mse_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(2),loglog(K,bias2_poly2,K,bias2_poly5,K,bias2_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(3),loglog(K,var_poly2,K,var_poly5,K,var_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(4),loglog(K,t_poly2,K,t_poly5,K,t_poly8)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('deg=2','deg=5','deg=8','spec')

%% N_o = 100
clear; clc
load('re_poly2_100.mat')
mse_poly2=mse;
bias2_poly2=bias2;
var_poly2=var;
t_poly2=t_tr+t_pr;
load('re_poly5_100.mat')
mse_poly5=mse;
bias2_poly5=bias2;
var_poly5=var;
t_poly5=t_tr+t_pr;
load('re_poly8_100.mat')
mse_poly8=mse;
bias2_poly8=bias2;
var_poly8=var;
t_poly8=t_tr+t_pr;

K_plot = logspace(1,7,7);

figure(1),loglog(K,mse_poly2,K,mse_poly5,K,mse_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('MSE')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(2),loglog(K,bias2_poly2,K,bias2_poly5,K,bias2_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Bias^2')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(3),loglog(K,var_poly2,K,var_poly5,K,var_poly8,...
K_plot,5.5e-2*K_plot.^(-2/3)/K_plot(1)^(-2/3),'k--',K_plot,...
1.4e-2*K_plot.^(-1)/K_plot(1)^(-1),'k-.')
xlabel('Total number of inner stage samples k')
ylabel('Variance')
axis([10, 1e7, 1e-8, 1e3])
legend('deg=2','deg=5','deg=8','spec')

figure(4),loglog(K,t_poly2,K,t_poly5,K,t_poly8)
xlabel('Total number of inner stage samples k')
ylabel('time (s)')
legend('deg=2','deg=5','deg=8','spec')