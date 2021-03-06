#import EX1Bc as EX1
#K = [ii**5 for ii in range(2,23)]
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly2, filename='re_poly2_1')
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly5, filename='re_poly5_1')
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly8, filename='re_poly8_1')

#import EX10
#K = [ii**5 for ii in range(2,23)]
#EX10.conv(K,N_i=1,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_1')
#EX10.conv(K,N_i=10,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_10')
#EX10.conv(K,N_i=100,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_100')

import EX100V2c as EX100
K = [ii**5 for ii in range(2,20)]
#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_poly2,filename='re_poly2_1')
#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_poly5,filename='re_poly5_1')
#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_poly8,filename='re_poly8_1')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_poly2,filename='re_poly2_10')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_poly5,filename='re_poly5_10')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_poly8,filename='re_poly8_10')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_poly2,filename='re_poly2_100')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_poly5,filename='re_poly5_100')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_poly8,filename='re_poly8_100')

#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_ridge2,filename='re_ridge2_1')
#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_ridge5,filename='re_ridge5_1')
#EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_ridge8,filename='re_ridge8_1')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_ridge2,filename='re_ridge2_10')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_ridge5,filename='re_ridge5_10')
#EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_ridge8,filename='re_ridge8_10')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_ridge2,filename='re_ridge2_100')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_ridge5,filename='re_ridge5_100')
#EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_ridge8,filename='re_ridge8_100')


EX100.conv(K,N_i=1,L=100,regr_method=EX100.re_ridge15,filename='re_ridge15_1')
EX100.conv(K,N_i=10,L=100,regr_method=EX100.re_ridge15,filename='re_ridge15_10')
EX100.conv(K,N_i=100,L=100,regr_method=EX100.re_ridge15,filename='re_ridge15_100')
