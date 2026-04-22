import torch
import numpy as np
import pandapower.networks as pn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

CONFIG = {
    'data_path': "ieee39_N-1.npz",
    'batch_size': 64,
    'in_len': 24,
    'out_len': 1,
    'split': [0.7, 0.15, 0.15]
}

class IEEE39_Dataset(Dataset):
    def __init__(self, data, pb, c_gen, p_gen, indices, in_len, out_len):
        self.data, self.pb, self.c_gen, self.p_gen = data, pb, c_gen, p_gen
        self.indices, self.in_len, self.out_len = indices, in_len, out_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        # 1. 历史 5 维 (P, Q, V, C, Fault)
        x_hist = self.data[t: t + self.in_len]
        # 2. 当前物理状态 3 维 (P, Q, V)
        x_now = self.data[t + self.in_len, :, :3]
        # 3. 预测目标：提取 碳势(索引3) 与 故障标签(索引4)
        y = self.data[t + self.in_len: t + self.in_len + self.out_len, :, 3:5]

        pb_now = self.pb[t + self.in_len]
        c_gen_now = self.c_gen[t + self.in_len]
        p_gen_now = self.p_gen[t + self.in_len]

        return (torch.FloatTensor(x_hist), torch.FloatTensor(x_now), torch.FloatTensor(y),
                torch.FloatTensor(pb_now), torch.FloatTensor(c_gen_now), torch.FloatTensor(p_gen_now))

def load_data():
    path = CONFIG['data_path']
    print(f">>> 正在加载数据集: {path} ...")
    raw_file = np.load(path)
    raw_data = raw_file['data']
    raw_pb, raw_c_gen, raw_p_gen = raw_file['pb'], raw_file['c_gen'], raw_file['p_gen']

    total_len, n_nodes, num_features = raw_data.shape # num_features 此时是 5
    train_size = int(total_len * CONFIG['split'][0])

    scaler = StandardScaler()
    # 【重点规避】：仅对前 4 维 (P,Q,V,C) 进行展平与归一化
    train_slice = raw_data[:train_size, :, :4].reshape(train_size, -1)
    scaler.fit(train_slice)

    data_norm = np.copy(raw_data)
    full_data_to_scale = raw_data[:, :, :4].reshape(total_len, -1)
    scaled_features = scaler.transform(full_data_to_scale).reshape(total_len, n_nodes, 4)
    # 将归一化后的数据塞回去，第 5 维 (Fault) 维持 0.0/1.0
    data_norm[:, :, :4] = scaled_features

    train_idx = np.arange(0, train_size - CONFIG['in_len'] - CONFIG['out_len'])
    val_idx = np.arange(train_size, train_size + int(total_len * CONFIG['split'][1]) - CONFIG['in_len'] - CONFIG['out_len'])
    test_idx = np.arange(train_size + int(total_len * CONFIG['split'][1]), total_len - CONFIG['in_len'] - CONFIG['out_len'])

    train_loader = DataLoader(IEEE39_Dataset(data_norm, raw_pb, raw_c_gen, raw_p_gen, train_idx, CONFIG['in_len'], CONFIG['out_len']), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(IEEE39_Dataset(data_norm, raw_pb, raw_c_gen, raw_p_gen, val_idx, CONFIG['in_len'], CONFIG['out_len']), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(IEEE39_Dataset(data_norm, raw_pb, raw_c_gen, raw_p_gen, test_idx, CONFIG['in_len'], CONFIG['out_len']), batch_size=CONFIG['batch_size'], shuffle=False)

    net = pn.case39()
    lines_from, lines_to = [int(i) for i in net.line.from_bus], [int(i) for i in net.line.to_bus]
    trafos_from, trafos_to = [int(i) for i in net.trafo.hv_bus], [int(i) for i in net.trafo.lv_bus]
    edge_index = torch.tensor([lines_from + trafos_from, lines_to + trafos_to], dtype=torch.long)

    return train_loader, val_loader, test_loader, edge_index, scaler