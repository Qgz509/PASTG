import pandapower as pp
import pandapower.networks as pn
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

class IEEE39_DataGenerator:
    def __init__(self):
        print(">>> 正在初始化 IEEE 39 节点系统...")
        self.net = pn.case39()
        self.num_nodes = 39

        load_idx_39 = self.net.load[self.net.load.bus == 38].index
        if not load_idx_39.empty:
            p_shift = self.net.load.loc[load_idx_39, 'p_mw'].sum()
            q_shift = self.net.load.loc[load_idx_39, 'q_mvar'].sum()
            self.net.load.loc[load_idx_39, 'p_mw'] = 0.0
            self.net.load.loc[load_idx_39, 'q_mvar'] = 0.0
            load_idx_3 = self.net.load[self.net.load.bus == 2].index
            if not load_idx_3.empty:
                self.net.load.loc[load_idx_3, 'p_mw'] += p_shift
                self.net.load.loc[load_idx_3, 'q_mvar'] += q_shift

        self.net.ext_grid['emission_factor'] = 850.0
        self.rows_coal = [0, 1]
        self.rows_gas = [2, 3, 5]
        self.rows_wind = [4, 8]
        self.rows_solar = [6, 7]

        self.net.gen['emission_factor'] = 0.0
        self.net.gen['p_mw_base'] = self.net.gen.p_mw.copy()

        self.net.gen.loc[self.rows_coal, 'emission_factor'] = 850.0
        self.net.gen.loc[self.rows_gas, 'emission_factor'] = 450.0

        self.net.load['p_mw_base'] = self.net.load.p_mw.copy()
        self.net.load['q_mvar_base'] = self.net.load.q_mvar.copy()
        self.last_wind_speed = 0.8

    def get_continuous_factors(self, t):
        hour = t % 24
        load_f = (0.6 + 0.3 * np.exp(-((hour - 9) ** 2) / 5) + 0.4 * np.exp(
            -((hour - 19) ** 2) / 5)) * np.random.normal(1, 0.01)
        solar_f = np.exp(-((hour - 12) ** 2) / 8) if 6 <= hour <= 18 else 0.0
        trend = 0.8 + 0.3 * np.cos(2 * np.pi * (hour - 3) / 24)
        wind_f = np.clip(0.9 * self.last_wind_speed + 0.1 * trend + np.random.normal(0, 0.05), 0.2, 1.1)
        self.last_wind_speed = wind_f
        return load_f, solar_f, wind_f

    def calculate_carbon_flow(self):
        net = self.net
        n = self.num_nodes
        P_gen = np.zeros(n)
        C_gen = np.zeros(n)

        for i, row in net.gen.iterrows():
            if row.in_service:
                p = net.res_gen.at[i, 'p_mw']
                if p > 0:
                    bus = int(row.bus)
                    P_gen[bus] += p
                    C_gen[bus] += p * row.emission_factor

        if not net.res_ext_grid.empty:
            for i, row in net.ext_grid.iterrows():
                p = net.res_ext_grid.at[i, 'p_mw']
                if p > 0:
                    bus = int(row.bus)
                    P_gen[bus] += p
                    C_gen[bus] += p * row.emission_factor

        PB = np.zeros((n, n))
        for df, topo, f_c, t_c, pf_c, pt_c in [
            (net.res_line, net.line, 'from_bus', 'to_bus', 'p_from_mw', 'p_to_mw'),
            (net.res_trafo, net.trafo, 'hv_bus', 'lv_bus', 'p_hv_mw', 'p_lv_mw')
        ]:
            for i, row in df.iterrows():
                if not topo.at[i, 'in_service']: continue
                try:
                    f, t = int(topo.at[i, f_c]), int(topo.at[i, t_c])
                    pf, pt = row[pf_c], row[pt_c]
                    if pf > 0:
                        PB[t, f] += abs(pt)
                    else:
                        PB[f, t] += abs(pf)
                except:
                    continue

        P_sum = P_gen + PB.sum(axis=1)
        P_sum[P_sum < 1e-5] = 1.0

        try:
            c_val = spsolve(diags(P_sum) - csc_matrix(PB), C_gen)
            return c_val, PB, C_gen, P_gen
        except:
            return np.zeros(n), np.zeros((n, n)), np.zeros(n), np.zeros(n)

    def generate_dataset(self, hours=8760):
        print(f">>> 开始生成 {hours} 小时运行数据 (包含 N-1 突发故障)...")
        data, pb_data, c_gen_data, p_gen_data = [], [], [], []

        gen_indices = list(self.net.gen.index)
        line_indices = list(self.net.line.index)
        cnt_normal, cnt_fault, cnt_collapse = 0, 0, 0

        for t in tqdm(range(hours), desc="时序潮流仿真进度", ncols=100):
            fl, fs, fw = self.get_continuous_factors(t)

            self.net.load.p_mw = self.net.load.p_mw_base * fl
            self.net.load.q_mvar = self.net.load.q_mvar_base * fl
            self.net.gen.loc[self.rows_solar, 'p_mw'] = self.net.gen.loc[self.rows_solar, 'p_mw_base'] * fs * 1.2
            self.net.gen.loc[self.rows_wind, 'p_mw'] = self.net.gen.loc[self.rows_wind, 'p_mw_base'] * fw * 1.2

            # === N-1 随机故障注入与【精准定位】 ===
            rand_val = random.random()
            fault_gen_idx, fault_line_idx, is_fault = None, None, False
            fault_val = np.zeros(self.num_nodes) # 初始化 0 矩阵

            if rand_val < 0.02:
                fault_gen_idx = random.choice(gen_indices)
                self.net.gen.at[fault_gen_idx, 'in_service'] = False
                is_fault = True
                bus = int(self.net.gen.at[fault_gen_idx, 'bus'])
                fault_val[bus] = 1.0  # 精准标记脱网节点

            elif rand_val < 0.07:
                fault_line_idx = random.choice(line_indices)
                self.net.line.at[fault_line_idx, 'in_service'] = False
                is_fault = True
                from_bus = int(self.net.line.at[fault_line_idx, 'from_bus'])
                to_bus = int(self.net.line.at[fault_line_idx, 'to_bus'])
                fault_val[from_bus] = 1.0 # 精准标记线路两端节点
                fault_val[to_bus] = 1.0

            try:
                pp.runpp(self.net, algorithm='nr', max_iteration=40)
                if not self.net.converged or self.net.res_bus.vm_pu.min() < 0.6 or np.isnan(self.net.res_bus.vm_pu.values).any():
                    raise ValueError("Grid Collapsed")

                c_val, pb_matrix, c_gen_vec, p_gen_vec = self.calculate_carbon_flow()
                p_val = -self.net.res_bus.p_mw.values
                q_val = -self.net.res_bus.q_mvar.values
                v_val = self.net.res_bus.vm_pu.values

                # 拼接为 5 维特征
                res = np.stack([p_val, q_val, v_val, c_val, fault_val], axis=1)

                data.append(res)
                pb_data.append(pb_matrix)
                c_gen_data.append(c_gen_vec)
                p_gen_data.append(p_gen_vec)

                if is_fault:
                    cnt_fault += 1
                else:
                    cnt_normal += 1

            except:
                cnt_collapse += 1
                res = np.zeros((39, 5)) # 崩溃时 5 维
                res[:, 4] = 1.0         # 全网瘫痪标志
                data.append(res)
                pb_data.append(np.zeros((39, 39)))
                c_gen_data.append(np.zeros(39))
                p_gen_data.append(np.zeros(39))

            finally:
                if fault_gen_idx is not None: self.net.gen.at[fault_gen_idx, 'in_service'] = True
                if fault_line_idx is not None: self.net.line.at[fault_line_idx, 'in_service'] = True

        final_data = np.nan_to_num(np.array(data), nan=0.0)
        final_pb = np.nan_to_num(np.array(pb_data), nan=0.0)
        final_c_gen = np.nan_to_num(np.array(c_gen_data), nan=0.0)
        final_p_gen = np.nan_to_num(np.array(p_gen_data), nan=0.0)

        save_path = "ieee39_N-1.npz"
        np.savez_compressed(save_path, data=final_data, pb=final_pb, c_gen=final_c_gen, p_gen=final_p_gen)

        print("\n" + "=" * 40)
        print("📊 数据集生成报告:")
        print(f"   [1] 基础特征矩阵 shape: {final_data.shape} (P,Q,V,C,Fault)")
        print(f"   [✔] 文件已保存至: {save_path}")
        print("=" * 40 + "\n")

if __name__ == "__main__":
    generator = IEEE39_DataGenerator()
    generator.generate_dataset(hours=8760)