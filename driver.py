import EX10
K = [ii**5 for ii in range(2,23)]
EX10.conv(K,N_i=10,L=100,regr_method=re_poly8,filename='re_poly8_10_qmc')
EX10.conv(K,N_i=1,L=100,regr_method=re_poly15,filename='re_poly15_1_qmc')
EX10.conv(K,N_i=10,L=100,regr_method=re_ridge15,filename='re_ridge15_10_qmc')

import EX1Bc as EX1
K = [ii**5 for ii in range(2,23)]
EX1.conv(K, N_i=1, L=1000, regr_method=re_poly2, filename='re_poly2_1')
EX1.conv(K, N_i=1, L=1000, regr_method=re_poly5, filename='re_poly5_1')
EX1.conv(K, N_i=1, L=1000, regr_method=re_poly8, filename='re_poly8_1')
EX1.conv(K, N_i=1, L=1000, regr_method=re_spec, filename='re_spec_1')
EX1.conv(K, N_i=1, L=1000, regr_method=re_spec_full, filename='re_spec_full_1')