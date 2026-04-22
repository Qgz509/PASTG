import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

from data_loader import load_data, CONFIG
from model_PASTG import get_model

RESULT_DIR = 'result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

TRAIN_CONFIG = {
    'epochs': 200,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gamma': 500.0,      # 物理守恒权重
    'fault_weight': 5.0, # 故障节点惩罚放大倍数
    'patience': 15,
    'save_path': os.path.join(RESULT_DIR, 'best_PASTG_model.pth')
}

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, scaler, gamma=0.1, fault_weight=5.0):
        super(PhysicsGuidedLoss, self).__init__()
        self.gamma = gamma
        self.scaler = scaler
        self.fault_weight = fault_weight

    def forward(self, pred, target, pb_matrix, c_gen, p_gen):
        # pred: (B, 1, N, 1) -> 碳浓度
        # target: (B, 1, N, 2) -> [碳浓度, 故障标签]
        pred_c = pred[:, 0, :, 0]
        target_c = target[:, 0, :, 0]
        target_fault = target[:, 0, :, 1] # 提取精准定位标签

        # 1. 故障加权数据损失
        base_mse = (pred_c - target_c) ** 2
        # 故障节点权重为 (1 + fault_weight)，正常节点为 1
        weight_mask = 1.0 + target_fault * self.fault_weight
        loss_data = torch.mean(base_mse * weight_mask)

        if self.gamma <= 0:
            return loss_data, loss_data.item(), 0.0

        # 2. 物理守恒损失
        c_mean = torch.tensor(self.scaler.mean_[3::4], device=pred.device).float()
        c_scale = torch.tensor(self.scaler.scale_[3::4], device=pred.device).float()
        pred_c_real = torch.relu(pred_c * c_scale + c_mean)

        pb_pu, c_gen_pu, p_gen_pu = pb_matrix / 100.0, c_gen / 100.0, p_gen / 100.0
        p_sum = pb_pu.sum(dim=2) + p_gen_pu
        p_sum[p_sum < 1e-5] = 1.0

        output_carbon = p_sum * pred_c_real
        input_carbon = c_gen_pu + torch.bmm(pb_pu, pred_c_real.unsqueeze(-1)).squeeze(-1)
        loss_cons = torch.mean(((output_carbon - input_carbon) / 850.0) ** 2)

        return loss_data + self.gamma * loss_cons, loss_data.item(), loss_cons.item()

def inverse_transform(pred_tensor, true_tensor, scaler):
    p_np = pred_tensor.cpu().numpy()
    t_np = true_tensor.cpu().numpy()
    c_mean = scaler.mean_[3::4].reshape(1, -1)
    c_scale = scaler.scale_[3::4].reshape(1, -1)
    # true_tensor 索引 0 是碳浓度
    p_inv_c = p_np[:, 0, :, 0] * c_scale + c_mean
    t_inv_c = t_np[:, 0, :, 0] * c_scale + c_mean
    return p_inv_c, t_inv_c

def calculate_physics_residual(model, test_loader, scaler, edge_index):
    model.eval()
    total_residual, total_samples = 0.0, 0
    device = TRAIN_CONFIG['device']
    c_mean = torch.tensor(scaler.mean_[3::4], device=device).float()
    c_scale = torch.tensor(scaler.scale_[3::4], device=device).float()

    with torch.no_grad():
        for batch in test_loader:
            x_h, x_n, y, pb, cg, pg = [b.to(device) for b in batch]
            pred = model(x_h, x_n, edge_index)
            pred_c_real = torch.relu(pred[:, 0, :, 0] * c_scale + c_mean)

            pb_pu, cg_pu, pg_pu = pb / 100.0, cg / 100.0, pg / 100.0
            p_sum = pb_pu.sum(dim=2) + pg_pu
            p_sum[p_sum < 1e-5] = 1.0

            input_carbon = cg_pu + torch.bmm(pb_pu, pred_c_real.unsqueeze(-1)).squeeze(-1)
            residual = torch.abs(p_sum * pred_c_real - input_carbon)
            total_residual += residual.sum().item()
            total_samples += residual.numel()

    return total_residual / total_samples

def train():
    print(f"\n>>> 启动精准定位加权 PASTG 训练...")
    train_loader, val_loader, test_loader, edge_index, scaler = load_data()
    edge_index = edge_index.to(TRAIN_CONFIG['device'])

    model = get_model().to(TRAIN_CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'], weight_decay=TRAIN_CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    criterion = PhysicsGuidedLoss(scaler=scaler, gamma=TRAIN_CONFIG['gamma'], fault_weight=TRAIN_CONFIG['fault_weight'])

    best_val_loss = float('inf')
    patience_cnt = 0

    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        train_ls, data_ls, cons_ls = [], [], []
        for batch in train_loader:
            x_h, x_n, y, pb, cg, pg = [b.to(TRAIN_CONFIG['device']) for b in batch]
            optimizer.zero_grad()
            pred = model(x_h, x_n, edge_index)
            loss, l_data, l_cons = criterion(pred, y, pb, cg, pg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_ls.append(loss.item()); data_ls.append(l_data); cons_ls.append(l_cons)

        model.eval()
        val_ls = []
        with torch.no_grad():
            for batch in val_loader:
                x_h, x_n, y, pb, cg, pg = [b.to(TRAIN_CONFIG['device']) for b in batch]
                pred = model(x_h, x_n, edge_index)
                _, l_data, _ = criterion(pred, y, pb, cg, pg)
                val_ls.append(l_data)

        avg_val = np.mean(val_ls)
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:03d} | Val MSE: {avg_val:.4f} | Physics: {np.mean(cons_ls):.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val; patience_cnt = 0
            torch.save(model.state_dict(), TRAIN_CONFIG['save_path'])
        else:
            patience_cnt += 1
            if patience_cnt >= TRAIN_CONFIG['patience']: break

    model.load_state_dict(torch.load(TRAIN_CONFIG['save_path']))
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_h, x_n, y, _, _, _ = [b.to(TRAIN_CONFIG['device']) for b in batch]
            all_p.append(model(x_h, x_n, edge_index)); all_y.append(y)

    c_pred, c_true = inverse_transform(torch.cat(all_p), torch.cat(all_y), scaler)
    print(f"\n最终测试集指标: RMSE: {np.sqrt(mean_squared_error(c_true.flatten(), c_pred.flatten())):.4f}")

if __name__ == "__main__":
    train()