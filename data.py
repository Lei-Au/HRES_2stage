import numpy as np
import pandas as pd

data = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Total/Tdata.xlsx')
#data = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Spring/SPdata.xlsx')
#data = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Summer/SMdata.xlsx')
#data = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Autumn/ATdata.xlsx')
#data = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Winter/WTdata.xlsx')

res_load = np.array(data[["负载"]]).T[0]
v = np.array(data[["风"]])
PV_sr = np.array(data[["光"]])
PV_Ta = np.array(data[["温度"]])

rand_PV = np.array(data[["A"]])
rand_WT = np.array(data[["B"]])
rand_load = np.array(data[["C"]])

data_EV = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Total/TEVdata.xlsx')
#data_EV = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Spring/SPEVdata.xlsx')
#data_EV = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Summer/SMEVdata.xlsx')
#data_EV = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Autumn/ATEVdata.xlsx')
#data_EV = pd.read_excel('/Users/lei/Desktop/HRES_BP/data/Winter/WTEVdata.xlsx')

EV_arr = np.array(data_EV[["arr_time"]], dtype=object)
EV_dep = np.array(data_EV[["dep_time"]], dtype=object)
EV_initsoc = np.array(data_EV[["iniSOC"]], dtype=object)
EV_sign = np.array(data_EV[["sign"]], dtype=object)

data_input = [res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_WT, rand_load]
data_input_PV = [res_load, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_load]
data_input_PV_BT = [res_load, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_load]

