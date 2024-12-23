import numpy as np
import time
import copy
import pandas as pd

path = '/Users/lei/Desktop/HRES_BP/basic/'

data = pd.read_excel(path + 'data.xlsx')
res_load = np.array(data[["负载"]]).T[0]
v = np.array(data[["风"]])
PV_sr = np.array(data[["光"]])
PV_Ta = np.array(data[["温度"]])

rand_PV = np.array(data[["A"]])
rand_WT = np.array(data[["B"]])
rand_load = np.array(data[["C"]])

data_EV = pd.read_excel(path + 'datatotal.xlsx')
EV_arr = np.array(data_EV[["arr_time"]], dtype=object)
EV_dep = np.array(data_EV[["dep_time"]], dtype=object)
EV_initsoc = np.array(data_EV[["iniSOC"]], dtype=object)
EV_sign = np.array(data_EV[["sign"]], dtype=object)

EV_end = np.array(pd.read_excel(path + 'ev3.xlsx')[["EV_end"]], dtype=object).T[0]

data_input = [res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_WT, rand_load, EV_end]

Dimention = 3
# 函数 目标个数
Func_num = 2

def Func(data_input, output_sign, X):
    #res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_WT, rand_load
    Y = data_input
    hres = EVHres(X[0], X[1], X[2], Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7],Y[8], Y[9], Y[10], Y[11], output_sign)
    return hres.hresrun()

# PV_N, WT_N, BT_N, DG_N are decision variables

class EVHres:
    def __init__(self, PV_N, WT_N, BT_N, res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_ini, EV_sign,rand1,rand2,rand3, EV_end,  output_sign):
        self.PV_N = PV_N
        self.WT_N = WT_N
        self.BT_N = BT_N

        self.output_sign = output_sign

        self.T = 366 * 24  # total time (+1 day for rolling)
        self.res_load = res_load  # res_load

        self.v = v  # wind speed
        self.WT_ic = 13500
        self.WT_ac = 420

        self.PV_sr = PV_sr  # slolar radiation(24h)
        self.PV_Ta = PV_Ta
        self.PV_ic = 1070/5 #1 m^2 PV = 200W, 1070 is 1kW cost
        self.PV_ac = 13/5

        self.PV_WT_lifetime = 20

        self.BT_ic = 132*6
        self.BT_ac = 2.64*6
        self.BT_rc = 132*6
        self.BT_lifetime = 5
        # BT lifetime 5 yr
        # cost cite by Jia, K., Liu, C., Li, S., & Jiang, D. (2023). Modeling and optimization of a hybrid renewable
        # energy system integrated with gas turbine and energy storage. Energy Conversion and Management, 279, 116763.
        # the salvage values 5%

        self.salvage = 0.05 #salvage value
        self.discR = 0.05 #discount rate
        #disCR cite by Li, B., Roche, R., & Miraoui, A. (2017). Microgrid sizing with combined evolutionary algorithm
        # and MILP unit commitment. Applied energy, 188, 547-562.

        self.P_BTc_max = self.BT_N * 1 /0.95
        self.P_BTd_max = self.BT_N * 1.5 * 0.95
        #self.Cap_BT = self.BT_N * 6
        self.Cap_BT = 100

        self.EV_dep = EV_dep
        self.EV_arr = EV_arr
        self.EV_ini = EV_ini
        self.EV_sign = EV_sign #Cap sign 32 70

        self.price_init = np.array([0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 1.19, 1.19, 1.65, 1.65, 1.65, 1.19, 1.19, 1.19, 1.19, 1.19, 1.19, 1.19,
             1.65, 1.91, 1.91, 1.19, 0.84, 0.84])

        self.CP = 20 #charge pile number

        self.rand1 = rand1.T[0]
        self.rand2 = rand2.T[0]
        self.rand3 = rand3.T[0]

        self.EV_end = EV_end

    def PV(self):
        PV_NOTC = 43
        PV_eta_r = 0.158
        PV_eta_pc = 1
        PV_NT = 0.005
        PV_Tref = 25
        Ppv = np.zeros(self.T)
        for i in range(self.T):
            if self.PV_N == 0:
                Ppv[i] = 0
            else:
                Tc = self.PV_Ta[i] + self.PV_sr[i] * (PV_NOTC - 20) / 800
                PV_eta = PV_eta_r * PV_eta_pc * (1 - PV_NT * (Tc - PV_Tref))
                Ppv[i] = PV_eta * self.PV_N * self.PV_sr[i] /1000
        return Ppv

    def WT(self):
        Vr = 11  # rate wind speed
        Pwtr = 10000  # wind rate power 10kW
        Vin = 2.5
        Vout = 45
        Pwt = np.zeros(self.T)
        if self.WT_N != 0 :
            for i in range(self.T):
                if self.v[i] < Vin or self.v[i] > Vout:
                    Pwt[i] = 0
                elif self.v[i] >= Vin and self.v[i] < Vr:
                    Pwt[i] = self.WT_N * Pwtr * (self.v[i] - Vin) / (Vr - Vin)
                else:
                    Pwt[i] = self.WT_N * Pwtr
        else:
            return Pwt
        return Pwt/1000

    def BT(self, bt_now, E):
        bt_after = bt_now
        BT_E_cd = 0
        if E == 0:
            bt_after = bt_now
            BT_E_cd = 0
        if E > self.P_BTc_max:
            bt_after = bt_now + 0.95 * self.P_BTc_max
            BT_E_cd = self.P_BTc_max
        if E < - self.P_BTd_max:
            bt_after = bt_now - self.P_BTd_max
            BT_E_cd = - self.P_BTd_max*0.95
        if E > 0 and E <= self.P_BTc_max:
            bt_after = bt_now + 0.95 * E
            BT_E_cd = E
        if E < 0 and E >= -self.P_BTd_max:
            bt_after = bt_now + E
            BT_E_cd = E*0.95
        if bt_after > self.Cap_BT:
            bt_after = self.Cap_BT
            BT_E_cd = (self.Cap_BT - bt_now)/0.95
        if bt_after < 0:
            bt_after = 0
            BT_E_cd = (0 - bt_now)*0.95
        return bt_after, BT_E_cd

    def hresrun(self):
        d = 0 #day
        EV_cmax = 8
        WT_supply_total = self.WT()
        PV_supply_total = self.PV()

        WT_supply_actual = self.rand1 * self.WT()
        PV_supply_actual = self.rand2 * self.PV()
        res_load_actual = self.rand3 * self.res_load

        EV_in_seq = [] #EV in scheduling queue
        EV_SOC_state = []

        BT_SOC_state = self.Cap_BT * 0.2 # BT initial 20%

        EV_scheduling_result = []
        BTc_schedulint_result = np.zeros(8760)
        BTd_schedulint_result = np.zeros(8760)
        P_grid_buy_result = np.zeros(8760)

        WLA_t = np.zeros(8760) # WLA each time
        EV_i = 0
        EV_end_result = np.zeros(19604)

        cost_g_buy = 0 # total cost buy power

        while d < 365:
            ts = 0 #time step 0-24
            while ts < 24:
                tt = d * 24 + ts

                remove_index = []
                for i in range(len(EV_in_seq)):
                    if tt >= self.EV_dep[EV_in_seq[i]] or EV_SOC_state[i] >= self.EV_end[i]:
                        remove_index.append(i)
                        EV_end_result[EV_in_seq[i]]= EV_SOC_state[i]
                remove_index.sort(reverse=True) #index reverse
                for index in remove_index:
                    del EV_in_seq[index]
                    del EV_SOC_state[index]

                while True:
                    if EV_i >= len(self.EV_dep):
                        break
                    if tt >= self.EV_arr[EV_i] and tt < self.EV_dep[EV_i] and (EV_i not in EV_in_seq):
                        EV_in_seq.append(EV_i)
                        EV_SOC_state.append(self.EV_ini[EV_i][0])
                        EV_scheduling_result.append([])
                    else:
                        break
                    EV_i += 1
                    if EV_i >= len(self.EV_dep):
                        break
                    elif self.EV_arr[EV_i] > tt:
                        break

                remain_time = []
                Cap_n = []
                EV_dep_soc = []
                for i in EV_in_seq:
                    remain_time.append(self.EV_dep[i][0] - tt)
                    Cap_n.append(self.EV_sign[i][0] * 43 + 32)
                    EV_dep_soc.append(self.EV_end[i])

                priority= np.array([EV_dep_soc[x] - EV_SOC_state[x] for x in range(len(EV_SOC_state))]) / np.array(remain_time)
                sorted_indices = np.argsort(priority)[::-1]

                EV_load = 0
                n_cp = 0
                ev_n = len(EV_in_seq)
                while n_cp <= self.CP and ev_n > 0:
                    evnow = EV_SOC_state[int(sorted_indices[n_cp])] * Cap_n[int(sorted_indices[n_cp])]
                    if EV_dep_soc[int(sorted_indices[n_cp])]*Cap_n[int(sorted_indices[n_cp])] - evnow >= EV_cmax:
                        EV_SOC_state[int(sorted_indices[n_cp])] = (evnow + EV_cmax)/Cap_n[int(sorted_indices[n_cp])]
                        EV_scheduling_result[int(sorted_indices[n_cp])].append(EV_cmax)
                        EV_load += EV_cmax
                    else:
                        evcharge = EV_dep_soc[int(sorted_indices[n_cp])]*Cap_n[int(sorted_indices[n_cp])] - evnow
                        EV_SOC_state[int(sorted_indices[n_cp])] = EV_dep_soc[int(sorted_indices[n_cp])]
                        EV_scheduling_result[int(sorted_indices[n_cp])].append(evcharge)
                        EV_load += evcharge
                    n_cp += 1
                    ev_n -= 1

                EV_load = EV_load/0.95
                P_supply_p = PV_supply_total[tt] + WT_supply_total[tt]
                load_p = self.res_load[tt]

                P_supply_a = PV_supply_actual[tt] + WT_supply_actual[tt]
                load_a = res_load_actual[tt]

                BT_SOC_state_init = copy.deepcopy(BT_SOC_state)

                BTc_tmp = 0
                BTd_tmp = 0

                if P_supply_a >= load_a + EV_load:
                    BT_SOC_state, BTc_tmp = self.BT(BT_SOC_state_init, (P_supply_a - load_a - EV_load))
                    BTc_schedulint_result[tt] = BTc_tmp
                else:
                    BT_SOC_state, BTd_tmp = self.BT(BT_SOC_state_init, (P_supply_a - load_a - EV_load))
                    BTd_schedulint_result[tt] = BTd_tmp
                '''
                BT_SOC_state, BT_tmp = self.BT(BT_SOC_state_init, min((P_supply_a - load_a - EV_load), BT_tmp))
                '''
                if P_supply_a - BTc_tmp > load_a + EV_load:
                    WLA_t[tt] = P_supply_a - load_a - EV_load - BTc_tmp
                if P_supply_a - BTd_tmp < load_a + EV_load:
                    P_g = load_a + EV_load - (P_supply_a - BTd_tmp)
                    P_grid_buy_result[tt] = P_g
                    cost_g_buy += P_g * self.price_init[ts]
                ts += 1
            d += 1

        AC = 0
        RC = 0
        E_use = 0
        IC = self.PV_ic * self.PV_N + self.WT_ic * self.WT_N + self.BT_ic * self.BT_N
        AC_tmp = self.PV_ac * self.PV_N + self.WT_ac * self.WT_N + self.BT_ac * self.BT_N
        for i in range(self.PV_WT_lifetime):
            AC += AC_tmp / ((1 + self.discR) ** (i + 1))
        for i in range(5, 16, 5):
            RC += self.BT_rc * self.BT_N / ((1 + self.discR) ** i)
        SV = IC * self.salvage + self.BT_ic * self.BT_N * self.salvage * 3

        E_use_firstyr = np.sum(PV_supply_actual) + np.sum(WT_supply_actual) - np.sum(WLA_t)
        for i in range(self.PV_WT_lifetime):
            E_use += E_use_firstyr / ((1 + self.discR) ** (i + 1))

        LCOE = (IC + AC + RC + SV) / E_use
        WLA = np.sum(WLA_t) / (np.sum(PV_supply_actual) + np.sum(WT_supply_actual))
        print('WLA', np.sum(WLA_t))
        print('supply', (np.sum(PV_supply_actual) + np.sum(WT_supply_actual)))

        if self.output_sign == 1:
            df_sys = pd.DataFrame({
                'BTc_schedulint_result': BTc_schedulint_result,
                'BTd_schedulint_result': BTd_schedulint_result,
                'P_grid_buy_result': P_grid_buy_result,
                'WLA_t': WLA_t,
                'cost_g_buy': cost_g_buy
            })
            df_sys.to_excel(path + 'sys_scheduling_result0.xlsx', index=False)

            df_ev = pd.DataFrame({
                'EV_end': EV_end_result
            })
            df_ev.to_excel(path + 'ev.xlsx', index=False)

        result = np.array([LCOE, cost_g_buy])
        return result


if __name__ == "__main__":

    start_time = time.time()
    X = [984, 95, 28]
    output_sign = 0
    result = Func(data_input, output_sign, X)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print('result', result)
    print(f"代码运行时间: {elapsed_time} 秒")



