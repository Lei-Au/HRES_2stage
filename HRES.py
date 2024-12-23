import numpy as np
import pandas as pd
from docplex.mp.model import Model
import time
import csv
import multiprocessing as mp
from functools import partial

# PV_N, WT_N, BT_N, INV_AD_N, INV_DA_N are decision variables

class EVHres:
    def __init__(self, PV_N, WT_N, BT_N, INV_AD_N, INV_DA_N, res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_ini, EV_sign, rand1, rand2,
                 rand3, output_sign):
        self.PV_N = PV_N
        self.WT_N = WT_N
        self.BT_N = BT_N
        self.INV_AD_N = INV_AD_N
        self.INV_DA_N = INV_DA_N

        self.output_sign = output_sign

        self.T = 366 * 24  # total time (+1 day for rolling)
        self.res_load = res_load  # res_load

        self.v = v  # wind speed
        self.WT_ic = 13500
        self.WT_ac = 420

        self.PV_sr = PV_sr  # slolar radiation(24h)
        self.PV_Ta = PV_Ta
        self.PV_ic = 1070 / 5  # 1 m^2 PV = 200W, 1070 is 1kW cost
        self.PV_ac = 13 / 5

        self.PV_WT_lifetime = 20

        self.BT_ic = 132 * 6
        self.BT_ac = 2.64 * 6
        self.BT_rc = 132 * 6
        self.BT_lifetime = 5

        self.INV_ic = 100
        self.INV_ac = 2
        self.INV_rc = 100
        self.INV_lifetime = 10
        # cost cite by Jia, K., Liu, C., Li, S., & Jiang, D. (2023). Modeling and optimization of a hybrid renewable
        # energy system integrated with gas turbine and energy storage. Energy Conversion and Management, 279, 116763.
        # the salvage values 5%

        self.salvage = 0.05  # salvage value
        self.discR = 0.05  # discount rate
        # disCR cite by Li, B., Roche, R., & Miraoui, A. (2017). Microgrid sizing with combined evolutionary algorithm
        # and MILP unit commitment. Applied energy, 188, 547-562.

        self.P_BTc_max = self.BT_N * 3
        self.P_BTd_max = self.BT_N * 3
        self.Cap_BT = self.BT_N * 6
        #C-rate: 0.5C

        self.EV_dep = EV_dep
        self.EV_arr = EV_arr
        self.EV_ini = EV_ini
        self.EV_sign = EV_sign

        '''
        #California TOU price
        self.price_init = np.array(
            [0.18867, 0.18867, 0.18867, 0.18867, 0.18867, 0.18867, 0.18867, 0.18867, 0.18867, 0.16201, 0.16201, 0.16201,
             0.16201, 0.16201, 0.18867, 0.18867, 0.38068, 0.38068, 0.38068, 0.38068, 0.38068, 0.18867, 0.18867,
             0.18867])
        '''

        # Shanghai TOU price
        self.price_init = np.array(
            [0.04239, 0.04239, 0.04239, 0.04239, 0.04239, 0.04239, 0.08803, 0.08803, 0.14282, 0.14282, 0.14282, 0.08803,
             0.08803, 0.08803, 0.08803, 0.08803, 0.08803, 0.08803, 0.14282, 0.14282, 0.14282, 0.08803, 0.04239,
             0.04239])

        self.CP = 10  # charge pile number

        self.rand1 = rand1.T[0]
        self.rand2 = rand2.T[0]
        self.rand3 = rand3.T[0]

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
                Ppv[i] = PV_eta * self.PV_N * self.PV_sr[i] / 1000
        return Ppv

    def WT(self):
        Vr = 11  # rate wind speed
        Pwtr = 10000  # wind rate power 10kW
        Vin = 2.5
        Vout = 45
        Pwt = np.zeros(self.T)
        if self.WT_N != 0:
            for i in range(self.T):
                if self.v[i] < Vin or self.v[i] > Vout:
                    Pwt[i] = 0
                elif self.v[i] >= Vin and self.v[i] < Vr:
                    Pwt[i] = self.WT_N * Pwtr * (self.v[i]**3 - Vin**3) / (Vr**3- Vin**3)
                else:
                    Pwt[i] = self.WT_N * Pwtr
        else:
            return Pwt
        return Pwt / 1000  # W --> kW

    def rolling_ev_opt(self, t_now, n, T_tmp, BT_init, EV_input, supply_input, price, EVC_sign):

        PV_supply = supply_input[0]
        WT_supply = supply_input[1]
        L_A_t = supply_input[2]

        EV_arr = [x - t_now for x in EV_input[0].tolist()]
        EV_dep = [x - t_now for x in EV_input[1]]
        EV_initSOC = EV_input[2]
        Cap_n = EV_input[3]

        c_ipt_sign = EVC_sign[0]
        last_time_sign = EVC_sign[1]

        P_EVc_max = 20 * np.ones(n)

        ru = 0.8
        pai_a = 1000
        pai_c = 100
        eta_BT = 0.95
        eta_AD = 0.95
        eta_EV = 0.95
        M = 9999999
        epsilon = 0.0001

        mdl = Model(name='HRES_BP_EV')

        P_EVc_up = np.tile(P_EVc_max[:, np.newaxis], T_tmp).flatten()
        E_EV_up = np.repeat(Cap_n, T_tmp).flatten()

        P_EVc_nt = mdl.continuous_var_matrix(n, T_tmp, lb=0, ub=P_EVc_up, name='P_EVc_nt')
        P_BTc_t = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTc_max, name='P_BTc_t')
        P_BTd_t = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTd_max, name='P_BTd_t')
        E_EV_nt = mdl.continuous_var_matrix(n, T_tmp, lb=0, ub=E_EV_up, name='E_EV_nt')
        E_BT_nt = mdl.continuous_var_list(T_tmp + 1, lb=0, ub=self.Cap_BT, name='E_BT_nt')

        P_PV_t = mdl.continuous_var_list(T_tmp, lb=0, ub=PV_supply.flatten(), name='P_PV_t')
        P_WT_t = mdl.continuous_var_list(T_tmp, lb=0, ub=WT_supply.flatten(), name='P_WT_t')

        P_g = mdl.continuous_var_list(T_tmp, lb=0, name='P_g')  # power buy from the grid
        z_a = mdl.continuous_var_list(T_tmp, lb=0, name='z_a')
        z_ad = mdl.continuous_var_list(T_tmp, lb=0, name='z_ad')
        z_d = mdl.continuous_var_list(T_tmp, lb=0, name='z_d')
        z_da = mdl.continuous_var_list(T_tmp, lb=0, name='z_da')

        x_EV_nt = mdl.binary_var_matrix(n, T_tmp, name='x_EV_nt')
        x_BT_t = mdl.binary_var_list(T_tmp, name='x_BT_t')
        y_BT_t = mdl.binary_var_list(T_tmp, name='y_BT_t')
        z_EV_nt = mdl.binary_var_matrix(n, T_tmp, name='z_EV_nt')

        C_BT = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTc_max, name='C_BT')
        D_BT = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTd_max, name='D_BT')

        sr = mdl.continuous_var_list(n, lb=0, name='sr')
        cr = mdl.continuous_var_list(n, lb=0, name='cr')

        # Constraints
        for t in range(T_tmp):
            mdl.add_constraint(z_a[t] + (1 / eta_AD) * z_ad[t] == P_g[t] + P_WT_t[t])
            mdl.add_constraint((1 / eta_AD) * z_da[t] + z_d[t] == eta_BT * D_BT[t] + P_PV_t[t])
            mdl.add_constraint(
                (1 / eta_BT) * C_BT[t] + (1 / eta_EV) * mdl.sum(P_EVc_nt[i, t] for i in range(n)) == z_d[t] + z_ad[t])
            mdl.add_constraint(L_A_t[t] == z_a[t] + z_da[t])

            mdl.add_constraint(z_ad[t] <= self.INV_AD_N)
            mdl.add_constraint(z_da[t] <= self.INV_DA_N)

            if t == 0:
                mdl.add_constraint(E_BT_nt[t + 1] == BT_init + C_BT[t] - D_BT[t])
            else:
                mdl.add_constraint(E_BT_nt[t + 1] == E_BT_nt[t] + C_BT[t] - D_BT[t])
            mdl.add_constraint(x_BT_t[t] + y_BT_t[t] <= 1)

            mdl.add_constraint(C_BT[t] <= self.P_BTc_max * x_BT_t[t])
            mdl.add_constraint(C_BT[t] <= P_BTc_t[t])
            mdl.add_constraint(C_BT[t] >= P_BTc_t[t] - self.P_BTc_max * (1 - x_BT_t[t]))

            mdl.add_constraint(D_BT[t] <= self.P_BTd_max * y_BT_t[t])
            mdl.add_constraint(D_BT[t] <= P_BTd_t[t])
            mdl.add_constraint(D_BT[t] >= P_BTd_t[t] - self.P_BTd_max * (1 - y_BT_t[t]))

            for i in range(n):
                if t >= EV_arr[i] and t < EV_dep[i][0]:
                    if t == EV_arr[i]:
                        mdl.add_constraint(E_EV_nt[i, t + 1] == EV_initSOC[i] * Cap_n[i] + P_EVc_nt[i, t])
                    else:
                        mdl.add_constraint(E_EV_nt[i, t + 1] == E_EV_nt[i, t] + P_EVc_nt[i, t])

                mdl.add_constraint(P_EVc_nt[i, t] - 0.000001 <= M * x_EV_nt[i, t])
                mdl.add_constraint(P_EVc_nt[i, t] >= 0.000001 - M * (1 - x_EV_nt[i, t]))

            mdl.add_constraint(mdl.sum(x_EV_nt[i, t] for i in range(n)) <= self.CP)

        for i in range(n):
            mdl.add_constraint(sr[i] >= pai_a * (ru * Cap_n[i] - E_EV_nt[i, EV_dep[i][0]]))

            mdl.add_constraint(z_EV_nt[i, int(EV_arr[i])] <= last_time_sign[i])  # t=arr-1 --> 0
            mdl.add_constraint(z_EV_nt[i, int(EV_arr[i])] <= x_EV_nt[i, int(EV_arr[i])])
            mdl.add_constraint(z_EV_nt[i, int(EV_arr[i])] >= last_time_sign[i] + x_EV_nt[i, int(EV_arr[i])] - 1)

            for t in range(int(EV_arr[i]) + 1, int(EV_dep[i][0])):
                mdl.add_constraint(z_EV_nt[i, t] <= x_EV_nt[i, t - 1])
                mdl.add_constraint(z_EV_nt[i, t] <= x_EV_nt[i, t])
                mdl.add_constraint(z_EV_nt[i, t] >= x_EV_nt[i, t - 1] + x_EV_nt[i, t] - 1)

            mdl.add_constraint(z_EV_nt[i, int(EV_dep[i][0])] <= x_EV_nt[i, int(EV_dep[i][0])-1])
            mdl.add_constraint(z_EV_nt[i, int(EV_dep[i][0])] <= 0)
            mdl.add_constraint(z_EV_nt[i, int(EV_dep[i][0])] >= x_EV_nt[i, int(EV_dep[i][0])-1] + 0 - 1)

            if c_ipt_sign[i] == 0:
                mdl.add_constraint(cr[i] >= pai_c * (0.5 * (
                        2 * mdl.sum(x_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]))) - 2 * mdl.sum(
                        z_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]) + 1))) - 1))
            elif c_ipt_sign[i] == 1:
                mdl.add_constraint(cr[i] >= pai_c * (0.5 * (
                        2 * mdl.sum(x_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]))) - 2 * mdl.sum(
                        z_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]) + 1))) - 0.5))
            elif c_ipt_sign[i] == 2:
                mdl.add_constraint(cr[i] >= pai_c * (0.5 * (
                        2 * mdl.sum(x_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]))) - 2 * mdl.sum(
                        z_EV_nt[i, j] for j in range(int(EV_arr[i]), int(EV_dep[i][0]) + 1))) ))

        # Objective function
        obj1 = mdl.sum(price * P_g)
        obj2 = mdl.sum(sr)
        obj3 = mdl.sum(cr)
        mdl.minimize(obj1 + obj2 + obj3)
        mdl.parameters.timelimit = 60
        # Solve the model
        try:
            solution = mdl.solve()
            # solution = mdl.solve(log_output=True)
            if solution:
                ##print('Solver thinks it is feasible')
                P_BTc_t = np.array(solution.get_value_list(P_BTc_t))
                P_BTd_t = np.array(solution.get_value_list(P_BTd_t))
                P_EVc_nt = np.array([[solution.get_value(P_EVc_nt[i, j]) for j in range(T_tmp)] for i in range(n)])

                P_PV_t = np.array(solution.get_value_list(P_PV_t))
                P_WT_t = np.array(solution.get_value_list(P_WT_t))
                E_EV_nt = np.array([[solution.get_value(E_EV_nt[i, j]) for j in range(T_tmp)] for i in range(n)])
                x_EV_nt = np.array([[solution.get_value(x_EV_nt[i, j]) for j in range(T_tmp)] for i in range(n)])
                E_BT_nt = np.array(solution.get_value_list(E_BT_nt))
                P_g = np.array(solution.get_value_list(P_g))

                z_a = np.array(solution.get_value_list(z_a))
                z_ad = np.array(solution.get_value_list(z_ad))
                z_d = np.array(solution.get_value_list(z_d))
                z_da = np.array(solution.get_value_list(z_da))
                C_BT = np.array(solution.get_value_list(C_BT))
                D_BT = np.array(solution.get_value_list(D_BT))

                results = [P_BTc_t, P_BTd_t, P_EVc_nt, P_g, P_PV_t, P_WT_t, E_EV_nt, E_BT_nt, x_EV_nt]
                opt = [z_a[0], z_ad[0], z_d[0], z_da[0], C_BT[0], D_BT[0], P_PV_t[0], P_WT_t[0], np.sum(P_EVc_nt, axis=0)[0]]
                return results, opt
            else:
                print('Solver thinks it is infeasible')
                results = 'error'
                return results
        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"An error occurred: {e}")

    def rolling_opt(self, T_tmp, BT_init, supply_input, price):

        PV_supply = supply_input[0]
        WT_supply = supply_input[1]
        L_A_t = supply_input[2]

        eta_BT = 0.95
        eta_AD = 0.95

        mdl = Model(name='HRES_BP')

        P_BTc_t = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTc_max, name='P_BTc_t')
        P_BTd_t = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTd_max, name='P_BTd_t')
        E_BT_nt = mdl.continuous_var_list(T_tmp + 1, lb=0, ub=self.Cap_BT, name='E_BT_nt')

        P_PV_t = mdl.continuous_var_list(T_tmp, lb=0, ub=PV_supply.flatten(), name='P_PV_t')
        P_WT_t = mdl.continuous_var_list(T_tmp, lb=0, ub=WT_supply.flatten(), name='P_WT_t')

        P_g = mdl.continuous_var_list(T_tmp, lb=0, name='P_g')  # power buy from the grid
        z_a = mdl.continuous_var_list(T_tmp, lb=0, name='z_a')
        z_ad = mdl.continuous_var_list(T_tmp, lb=0, name='z_ad')
        z_d = mdl.continuous_var_list(T_tmp, lb=0, name='z_d')
        z_da = mdl.continuous_var_list(T_tmp, lb=0, name='z_da')

        x_BT_t = mdl.binary_var_list(T_tmp, name='x_BT_t')
        y_BT_t = mdl.binary_var_list(T_tmp, name='y_BT_t')

        C_BT = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTc_max, name='C_BT')
        D_BT = mdl.continuous_var_list(T_tmp, lb=0, ub=self.P_BTd_max, name='D_BT')

        # Constraints
        for t in range(T_tmp):
            mdl.add_constraint(z_a[t] + (1 / eta_AD) * z_ad[t] == P_g[t] + P_WT_t[t])
            mdl.add_constraint((1 / eta_AD) * z_da[t] + z_d[t] == eta_BT * D_BT[t] + P_PV_t[t])
            mdl.add_constraint((1 / eta_BT) * C_BT[t] == z_d[t] + z_ad[t])
            mdl.add_constraint(L_A_t[t] == z_a[t] + z_da[t])

            mdl.add_constraint(z_ad[t] <= self.INV_AD_N)
            mdl.add_constraint(z_da[t] <= self.INV_DA_N)

            if t == 0:
                mdl.add_constraint(E_BT_nt[t + 1] == BT_init + C_BT[t] - D_BT[t])
            else:
                mdl.add_constraint(E_BT_nt[t + 1] == E_BT_nt[t] + C_BT[t] - D_BT[t])
            mdl.add_constraint(x_BT_t[t] + y_BT_t[t] <= 1)

            mdl.add_constraint(C_BT[t] <= self.P_BTc_max * x_BT_t[t])
            mdl.add_constraint(C_BT[t] <= P_BTc_t[t])
            mdl.add_constraint(C_BT[t] >= P_BTc_t[t] - self.P_BTc_max * (1 - x_BT_t[t]))

            mdl.add_constraint(D_BT[t] <= self.P_BTd_max * y_BT_t[t])
            mdl.add_constraint(D_BT[t] <= P_BTd_t[t])
            mdl.add_constraint(D_BT[t] >= P_BTd_t[t] - self.P_BTd_max * (1 - y_BT_t[t]))

        # Objective function
        obj = mdl.sum(price * P_g)
        mdl.minimize(obj)

        # Solve the model
        try:
            solution = mdl.solve()
            # solution = mdl.solve(log_output=True)
            if solution:
                ##print('Solver thinks it is feasible')
                P_BTc_t = np.array(solution.get_value_list(P_BTc_t))
                P_BTd_t = np.array(solution.get_value_list(P_BTd_t))

                P_PV_t = np.array(solution.get_value_list(P_PV_t))
                P_WT_t = np.array(solution.get_value_list(P_WT_t))
                E_BT_nt = np.array(solution.get_value_list(E_BT_nt))
                P_g = np.array(solution.get_value_list(P_g))

                z_a = np.array(solution.get_value_list(z_a))
                z_ad = np.array(solution.get_value_list(z_ad))
                z_d = np.array(solution.get_value_list(z_d))
                z_da = np.array(solution.get_value_list(z_da))
                C_BT = np.array(solution.get_value_list(C_BT))
                D_BT = np.array(solution.get_value_list(D_BT))

                results = [P_BTc_t, P_BTd_t, P_g, P_PV_t, P_WT_t, E_BT_nt]
                opt = [z_a[0], z_ad[0], z_d[0], z_da[0], C_BT[0], D_BT[0], P_PV_t[0], P_WT_t[0]]

                return results, opt
            else:
                print('Solver thinks it is infeasible')
                results = 'error'
                return results
        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"An error occurred: {e}")

    def hresrun(self):
        d = 0  # day

        WT_supply_total = self.WT()
        PV_supply_total = self.PV()

        PV_supply_actual = self.rand1 * self.PV()
        WT_supply_actual = self.rand2 * self.WT()
        res_load_actual = self.rand3 * self.res_load

        EV_in_seq = []  # EV in scheduling queue
        EV_SOC_state = []
        EV_nmd = [] # EV not meet demand, index/SOC

        BT_SOC_state = self.Cap_BT * 0.2  # BT initial 20%

        EV_scheduling_result = []
        BTc_schedulint_result = []
        BTd_schedulint_result = []
        P_grid_buy_result = []
        cd_sign_all = []

        WLA_t = []  # WLA each time
        EV_i = 0 # EV index

        flow = []
        EV_charging_t_sum = []

        #Initialize the maximum AC/DC conversion value
        AC_DC_max = 0
        DC_AC_max = 0

        cost_g_buy = 0  # total cost buy power

        while d < 365:
            print('size', [self.PV_N, self.WT_N, self.BT_N, self.INV_AD_N, self.INV_DA_N], 'd', d)
            ts = 0  # time step 0-24
            while ts < 24:
                tt = d * 24 + ts  ##time step 0-8760
                # print('size',[self.PV_N, self.WT_N, self.BT_N], 't',tt)
                # print('EV_in_seq',EV_in_seq)
                # print('EV_SOC_state', EV_SOC_state)

                remove_index = []
                for i in range(len(EV_in_seq)):
                    if tt >= self.EV_dep[EV_in_seq[i]] or EV_SOC_state[i] >= 0.799999:

                        if tt == self.EV_dep[EV_in_seq[i]] and EV_SOC_state[i] < 0.799:
                            EV_nmd.append([EV_i, EV_SOC_state[i]])

                        remove_index.append(i)
                remove_index.sort(reverse=True)  # index reverse
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
                        cd_sign_all.append([])
                    else:
                        break
                    EV_i += 1
                    if EV_i >= len(self.EV_dep):
                        break
                    elif self.EV_arr[EV_i] > tt:
                        break

                price = []
                Cap_n = []
                EV_dep_tmp = []
                last_cd_sign = []
                EV_arr_tmp = tt * np.ones(len(EV_in_seq))
                for i in EV_in_seq:
                    EV_dep_tmp.append(self.EV_dep[i])
                    ##########################################改Cap_EV
                    Cap_n.append(self.EV_sign[i][0] * 43 + 32)

                    if cd_sign_all[i] == []:
                        last_cd_sign.append(0)
                    else:
                        last_cd_sign.append(cd_sign_all[i][-1])

                EV_ipt = [EV_arr_tmp, EV_dep_tmp, EV_SOC_state, Cap_n]

                #print('cd_sign_all', cd_sign_all)
                #print('EV_scheduling_result',EV_scheduling_result)

                c_sign = []
                for i in range(len(EV_in_seq)):
                    if cd_sign_all[EV_in_seq[i]] == [] or sum(cd_sign_all[EV_in_seq[i]]) == 0:
                        c_time_sign = 0
                    else:
                        if last_cd_sign[i] == 0:
                            c_time_sign = 2
                        else:
                            trans = sum(1 for j in range(1, len(cd_sign_all[EV_in_seq[i]])) if cd_sign_all[EV_in_seq[i]][j] != cd_sign_all[EV_in_seq[i]][j - 1])
                            if trans > 1:
                                c_time_sign = 2
                            else:
                                c_time_sign = 1
                    c_sign.append(c_time_sign)

                EVc_sign = [c_sign, last_cd_sign]

                #print('EVc_sign', EVc_sign)

                if EV_in_seq == []:
                    T_tmp = 24
                else:
                    T_tmp = np.max(EV_dep_tmp) - tt + 1
                    if T_tmp < 24:
                        T_tmp = 24

                PV_supply_tmp = PV_supply_total[tt:(T_tmp + tt)]
                PV_supply_tmp[0] = PV_supply_actual[tt]

                WT_supply_tmp = WT_supply_total[tt:(T_tmp + tt)]
                WT_supply_tmp[0] = WT_supply_actual[tt]

                res_load_tmp = self.res_load[tt:(T_tmp + tt)]
                res_load_tmp[0] = res_load_actual[tt]

                supply_ipt = [PV_supply_tmp, WT_supply_tmp, res_load_tmp]

                for i in range(T_tmp):
                    price.append(self.price_init[(ts + i) % 24])

                price = np.array(price)

                # 更新EVsoc, BTsoc, powerbuy, WLA
                if EV_in_seq == []:
                    # RE, [P_BTc_t, P_BTd_t, P_g, P_PV_t, P_WT_t, E_BT_nt]
                    #         0        1      2     3        4       5
                    RE, flow_opt = self.rolling_opt(T_tmp, BT_SOC_state, supply_ipt, price)

                    if flow_opt[1] > AC_DC_max:
                        #AC-->DC
                        AC_DC_max = flow_opt[1]
                    if flow_opt[3] > DC_AC_max:
                        # DC-->AC
                        DC_AC_max = flow_opt[3]

                    flow.append(flow_opt)
                    EV_charging_t_sum.append(0)

                    BT_SOC_state = RE[5][1]
                    BTc_schedulint_result.append(RE[0][0])
                    BTd_schedulint_result.append(-RE[1][0])
                    P_grid_buy_result.append(RE[2][0])
                    WLA_t.append((WT_supply_actual[tt] + PV_supply_actual[tt] - RE[3][0] - RE[4][0]))
                    cost_g_buy += (RE[2][0] * self.price_init[ts])  # * price[0])
                else:
                    # RE, [P_BTc_t, P_BTd_t, P_EVc_nt, P_g, P_PV_t, P_WT_t, E_EV_nt, E_BT_nt, x_EV_nt]
                    #         0        1         2      3      4      5        6        7        8
                    #start_time = time.time()
                    RE, flow_opt = self.rolling_ev_opt(tt, len(EV_in_seq), T_tmp, BT_SOC_state, EV_ipt, supply_ipt, price, EVc_sign)
                    #end_time = time.time()
                    #elapsed_time = end_time - start_time
                    #print(tt, 'Run time:', elapsed_time)

                    if flow_opt[1] > AC_DC_max:
                        #AC-->DC
                        AC_DC_max = flow_opt[1]
                    if flow_opt[3] > DC_AC_max:
                        # DC-->AC
                        DC_AC_max = flow_opt[3]

                    flow.append(flow_opt[:-1])
                    EV_charging_t_sum.append(flow_opt[-1])

                    EV_SOC_state = [a / b for a, b in zip(RE[6][:, 1].tolist(), Cap_n)]  # RE[6][:,1].tolist()
                    ##print('EV_SOC_state', EV_SOC_state)
                    BT_SOC_state = RE[7][1]
                    RE_index_tmp = 0
                    for i in EV_in_seq:
                        EV_scheduling_result[i].append(RE[2][RE_index_tmp, 0])
                        if RE[2][RE_index_tmp, 0] > 0.0001:
                            cd_sign_all[i].append(1)
                        else:
                            cd_sign_all[i].append(0)
                        RE_index_tmp += 1

                    BTc_schedulint_result.append(RE[0][0])
                    BTd_schedulint_result.append(-RE[1][0])
                    P_grid_buy_result.append(RE[3][0])
                    WLA_t.append((WT_supply_actual[tt] + PV_supply_actual[tt] - RE[4][0] - RE[5][0]))
                    cost_g_buy += (RE[3][0] * self.price_init[ts])  # * price[0])
                ts += 1
            d += 1

        EV_nmd = np.array(EV_nmd)
        flow = np.array(flow)
        EV_charging_t_sum = np.array(EV_charging_t_sum)
        #print('EV_nmd', EV_nmd)

        #print AC\DC  CONVERSION
        print('AC_DC', [AC_DC_max, DC_AC_max])

        AC = 0
        RC = 0
        E_use = 0
        IC = self.PV_ic * self.PV_N + self.WT_ic * self.WT_N + self.BT_ic * self.BT_N + (self.INV_AD_N + self.INV_DA_N)*self.INV_ic
        AC_tmp = self.PV_ac * self.PV_N + self.WT_ac * self.WT_N + self.BT_ac * self.BT_N + (self.INV_AD_N + self.INV_DA_N)*self.INV_ac
        for i in range(self.PV_WT_lifetime):
            AC += AC_tmp / ((1 + self.discR) ** (i + 1))
        for i in range(5, 16, 5):
            RC += self.BT_rc * self.BT_N / ((1 + self.discR) ** i)
        RC += self.INV_rc * (self.INV_AD_N + self.INV_DA_N) / ((1 + self.discR) ** 10)
        SV = IC * self.salvage + self.BT_ic * self.BT_N * self.salvage * 3 + (self.INV_AD_N + self.INV_DA_N) * self.INV_ic * self.salvage

        E_use_firstyr = np.sum(PV_supply_actual) + np.sum(WT_supply_actual) - np.sum(WLA_t)
        for i in range(self.PV_WT_lifetime):
            E_use += E_use_firstyr / ((1 + self.discR) ** (i + 1))

        if E_use == 0:
            LCOE = 0
        else:
            LCOE = (IC + AC + RC - SV) / E_use
        # WLA = np.sum(WLA_t) / (np.sum(PV_supply_actual) + np.sum(WT_supply_actual))

        result = np.array([LCOE, cost_g_buy])

        if self.output_sign == 1:

            EVS_path = '/Users/lei/Desktop/HRES_BP/result/EV_scheduling_result.csv'
            with open(EVS_path, 'a+', newline='') as f:
                csv_write = csv.writer(f)
                for row in EV_scheduling_result:
                    csv_write.writerow(row)

            df_sys = pd.DataFrame({
                'BTc_schedulint_result': BTc_schedulint_result,
                'BTd_schedulint_result': BTd_schedulint_result,
                'P_grid_buy_result': P_grid_buy_result,
                'WLA_t': WLA_t,
                'cost_g_buy': cost_g_buy,
                'power_buy': power_buy
            })
            path = '/Users/lei/Desktop/HRES_BP/result/sys_scheduling_result.xlsx'
            df_sys.to_excel(path, index=False)

        return result


def Func(data_input, output_sign, X):
    # res_load, v, PV_sr, PV_Ta, EV_dep, EV_arr, EV_initsoc, EV_sign, rand_PV, rand_WT, rand_load
    Y = data_input
    hres = EVHres(X[0], X[1], X[2], X[3], X[4], Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7], Y[8], Y[9], Y[10], output_sign)
    return hres.hresrun()


'''
from data import data_input
if __name__ == "__main__":
    start_time = time.time()
    path = '/Users/lei/Desktop/HRES_BP/'
    data_N = pd.read_excel(path + 'result_all/N_PVWT.xlsx')
    # 遍历DataFrame的每一行
    X = []
    for index, row in data_N.iterrows():
        X.append(row.tolist())  # 打印每行的数据

    ops = 0

    fun_tmp = partial(Func, data_input, ops)
    pool = mp.Pool(processes=6)
    # pool = mp.Pool()

    ObjV = pool.map(fun_tmp, X)
    # ObjV = self.evaluate(self.data_input, ops, np.array([ind for ind in offspring if not ind.fitness.valid]))
    pool.close()
    pool.join()

    with open(path+'result_all/res.csv', 'a+', newline='') as f:
        csv_write = csv.writer(f)
        for row in np.array(ObjV):
            csv_write.writerow(row)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
'''