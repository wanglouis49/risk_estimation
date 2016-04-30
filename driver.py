import EX100Vc as EX100
reload(EX100)

port1 = EX100.EX100V()
port1.regr_data_prep(1000,10)
port1.poly_regr(2)
port1.svr(5000,1e-4)