#import EX1Bc as EX1
#K = [ii**5 for ii in range(2,23)]
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly2, filename='re_poly2_1')
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly5, filename='re_poly5_1')
#EX1.conv(K, N_i=1, L=1000, regr_method=EX10.re_poly8, filename='re_poly8_1')

import EX10
K = [ii**5 for ii in range(2,23)]
EX10.conv(K,N_i=1,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_1')
EX10.conv(K,N_i=10,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_10')
EX10.conv(K,N_i=100,L=100,regr_method=EX10.re_svrCV,filename='re_svrCV_100')