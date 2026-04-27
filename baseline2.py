import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import json

from data_loader import load_data, CONFIG
from model_PASTG import PositionalEncoding, GCNLayer, AdaptiveGraphLayer

# ==========================================
# 全局配置
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULT_DIR = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)

EPOCHS = 50
PATIENCE = 15
GAMMA_VAL = 500.0
FAULT_WEIGHT = 5.0

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 损失函数 (含故障加权与物理引导)
# ==========================================
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, scaler, gamma, fault_weight, use_fault_weight):
        super(PhysicsGuidedLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.scaler = scaler
        self.fault_weight = fault_weight
        self.use_fault_weight = use_fault_weight

    def forward(self, pred, target, pb_matrix, c_gen, p_gen):
        pred_c = pred[:, 0, :, 0]
        target_c = target[:, 0, :, 0]
        base_mse = self.mse_loss(pred_c, target_c)

        if self.use_fault_weight:
            target_fault = target[:, 0, :, 1]
            weight_mask = 1.0 + target_fault * self.fault_weight
            loss_data = torch.mean(base_mse * weight_mask)
        else:
            loss_data = torch.mean(base_mse)

        if self.gamma <= 0:
            return loss_data, loss_data.item(), 0.0

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


# ==========================================
# 2. 灵活可配置的模型架构 (PASTG & Baselines)
# ==========================================
class Flexible_TS_Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_nodes, config, dropout=0.1):
        super(Flexible_TS_Block, self).__init__()
        self.config = config
        if config['use_transformer']:
            self.temporal_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                      batch_first=True)
            self.norm_t = nn.LayerNorm(hidden_dim)
        elif config['use_gru']:
            self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.gcn_type = config['gcn_type']
        if self.gcn_type in ['physical', 'both']: self.gcn_static = GCNLayer(hidden_dim, hidden_dim)
        if self.gcn_type in ['adaptive', 'both']: self.gcn_adp = GCNLayer(hidden_dim, hidden_dim)
        if self.gcn_type == 'both': self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                                                     nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.norm_gcn = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_static, adj_adp):
        B, N, T, D = x.shape
        x_t_in = x.reshape(B * N, T, D)
        if self.config['use_transformer']:
            attn_out, _ = self.temporal_att(x_t_in, x_t_in, x_t_in)
            x_t_out = self.norm_t(x_t_in + attn_out)
        elif self.config['use_gru']:
            gru_out, _ = self.temporal_gru(x_t_in)
            x_t_out = gru_out
        else:
            x_t_out = x_t_in[:, -1:, :].expand(-1, T, -1)

        x = x_t_out.reshape(B, N, T, D)
        x_s_in = x.permute(0, 2, 1, 3).reshape(B * T, N, D)

        if self.gcn_type == 'none':
            x_s_out = x_s_in
        else:
            if self.gcn_type == 'physical':
                x_gcn_out = self.gcn_static(x_s_in, adj_static)
            elif self.gcn_type == 'adaptive':
                x_gcn_out = self.gcn_adp(x_s_in, adj_adp)
            elif self.gcn_type == 'both':
                out_s, out_a = self.gcn_static(x_s_in, adj_static), self.gcn_adp(x_s_in, adj_adp)
                gate = self.fusion_gate(torch.cat([out_s, out_a], dim=-1))
                x_gcn_out = gate * out_s + (1.0 - gate) * out_a
            x_s_out = self.norm_gcn(x_s_in + self.dropout(F.relu(x_gcn_out)))

        x = x_s_out.reshape(B, T, N, D).permute(0, 2, 1, 3)
        return self.norm_ff(x + self.feed_forward(x))


class FlexiblePASTG(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, out_len, config, hidden_dim=64):
        super(FlexiblePASTG, self).__init__()
        self.config, self.out_len, self.out_dim = config, out_len, out_dim
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.layer1 = Flexible_TS_Block(hidden_dim, 4, num_nodes, config)
        self.layer2 = Flexible_TS_Block(hidden_dim, 4, num_nodes, config)
        if config['gcn_type'] in ['adaptive', 'both']: self.adaptive_graph = AdaptiveGraphLayer(num_nodes, 20)
        head_in = hidden_dim + 3 if config['use_concat'] else hidden_dim
        self.output_head = nn.Sequential(nn.Linear(head_in, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, out_dim * out_len))

    def forward(self, x_hist, x_now, edge_index):
        B, T, N, F = x_hist.shape
        x = x_hist.permute(0, 2, 1, 3)
        adj = torch.eye(N).to(x.device)
        adj[edge_index[0], edge_index[1]] = 1.0;
        adj[edge_index[1], edge_index[0]] = 1.0
        d_inv = torch.pow(adj.sum(1), -0.5);
        d_inv[torch.isinf(d_inv)] = 0.
        norm_adj = torch.mm(torch.diag(d_inv), torch.mm(adj, torch.diag(d_inv)))
        adj_adp = self.adaptive_graph() if self.config['gcn_type'] in ['adaptive', 'both'] else None

        x = self.layer2(self.layer1(self.pos_encoder(self.embedding(x)), norm_adj, adj_adp), norm_adj, adj_adp)
        x_combined = torch.cat([x[:, :, -1, :], x_now], dim=-1) if self.config['use_concat'] else x[:, :, -1, :]
        return self.output_head(x_combined).reshape(B, N, self.out_len, self.out_dim).permute(0, 2, 1, 3)


class PureTransformer_Baseline(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, out_len, hidden_dim=64):
        super(PureTransformer_Baseline, self).__init__()
        self.num_nodes, self.out_len, self.out_dim = num_nodes, out_len, out_dim
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, out_dim * out_len))

    def forward(self, x_hist, x_now, edge_index):
        B, T, N, F = x_hist.shape
        x = self.pos_encoder(self.embedding(x_hist.permute(0, 2, 1, 3)))
        x = self.transformer_encoder(x.reshape(B * N, T, -1))
        out = self.output_head(x[:, -1, :])
        return out.reshape(B, N, self.out_len, self.out_dim).permute(0, 2, 1, 3)


# ==========================================
# 3. 实验核心逻辑
# ==========================================
def train_and_eval_model(exp_id, config):
    print(f"\n[{exp_id}/12] 运行中: {config['name']} ...")
    train_loader, val_loader, test_loader, edge_index, scaler = load_data()
    edge_index = edge_index.to(DEVICE)

    if exp_id == 3:
        model = PureTransformer_Baseline(num_nodes=39, in_dim=5, out_dim=1, out_len=CONFIG['out_len']).to(DEVICE)
    else:
        model = FlexiblePASTG(num_nodes=39, in_dim=5, out_dim=1, out_len=CONFIG['out_len'], config=config).to(DEVICE)

    if 'pretrained_path' in config and os.path.exists(config['pretrained_path']):
        model.load_state_dict(torch.load(config['pretrained_path'], map_location=DEVICE), strict=False)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = PhysicsGuidedLoss(scaler, GAMMA_VAL if config['use_carbon_loss'] else 0.0, FAULT_WEIGHT,
                                      config['use_fault_weight'])
        best_val_loss, patience_cnt = float('inf'), 0
        best_weights = None

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                x_h, x_n, y, pb, cg, pg = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                pred = model(x_h, x_n, edge_index)
                loss, _, _ = criterion(pred, y, pb, cg, pg)
                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0);
                optimizer.step()

            model.eval()
            val_ls = []
            with torch.no_grad():
                for batch in val_loader:
                    x_h, x_n, y, pb, cg, pg = [b.to(DEVICE) for b in batch]
                    _, l_d, _ = criterion(model(x_h, x_n, edge_index), y, pb, cg, pg)
                    val_ls.append(l_d)
            avg_val = np.mean(val_ls)
            if avg_val < best_val_loss:
                best_val_loss, best_weights, patience_cnt = avg_val, model.state_dict().copy(), 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break
        if best_weights: model.load_state_dict(best_weights)

    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_h, x_n, y, _, _, _ = [b.to(DEVICE) for b in batch]
            all_p.append(model(x_h, x_n, edge_index));
            all_y.append(y)

    p_np, t_np = torch.cat(all_p).cpu().numpy(), torch.cat(all_y).cpu().numpy()
    c_scale, c_mean = scaler.scale_[3::4].reshape(1, -1), scaler.mean_[3::4].reshape(1, -1)
    c_pred = np.maximum(p_np[:, 0, :, 0] * c_scale + c_mean, 0)
    c_true = t_np[:, 0, :, 0] * c_scale + c_mean

    y_t, y_p = c_true.flatten(), c_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)  # 新增 MAE 计算
    r2 = r2_score(y_t, y_p)

    # 物理残差计算逻辑保持
    def get_res():
        tr, ts = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                x_h, x_n, _, pb, cg, pg = [b.to(DEVICE) for b in batch]
                pr = torch.relu(model(x_h, x_n, edge_index)[:, 0, :, 0] * torch.tensor(c_scale,
                                                                                       device=DEVICE).float() + torch.tensor(
                    c_mean, device=DEVICE).float())
                pb_pu, cg_pu, pg_pu = pb / 100., cg / 100., pg / 100.
                p_s = pb_pu.sum(dim=2) + pg_pu;
                p_s[p_s < 1e-5] = 1.0
                out_c, in_c = p_s * pr, cg_pu + torch.bmm(pb_pu, pr.unsqueeze(-1)).squeeze(-1)
                res = torch.abs(out_c - in_c)
                tr += res.sum().item();
                ts += res.numel()
        return tr / ts

    residual = get_res()

    print(f"    [✔] 完成 -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Residual: {residual:.4f}")
    return rmse, mae, r2, residual


# ==========================================
# 4. 可视化模块 (支持 MAE 三柱图)
# ==========================================
def plot_metrics(names, rmses, maes, r2s, filename, title):
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(names))
    width = 0.25  # 三根柱子，宽度设为0.25

    rects1 = ax1.bar(x - width, rmses, width, label='RMSE', color='#1f77b4', alpha=0.8)  # 蓝色
    rects2 = ax1.bar(x, maes, width, label='MAE', color='#2ca02c', alpha=0.8)  # 绿色

    ax1.set_ylabel('Error (g/kWh)', color='black', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")

    for rect in rects1:
        ax1.annotate(f'{rect.get_height():.2f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#1f77b4')
    for rect in rects2:
        ax1.annotate(f'{rect.get_height():.2f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#2ca02c')

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, r2s, width, label='R² Score', color='#ff7f0e', alpha=0.8)  # 橙色
    ax2.set_ylabel('R² Score', color='#ff7f0e', fontweight='bold')
    ax2.set_ylim([max(0, min(r2s) - 0.1), 1.05])

    for rect in rects3:
        ax2.annotate(f'{rect.get_height():.3f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#ff7f0e')

    fig.suptitle(title, fontweight='bold', fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300);
    plt.close()


# ==========================================
# 5. 主运行流程
# ==========================================
EXPERIMENTS = {
    1: {"name": "Single Phy-GCN", "use_transformer": False, "use_gru": False, "gcn_type": "physical",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    2: {"name": "Single Adp-GCN", "use_transformer": False, "use_gru": False, "gcn_type": "adaptive",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    3: {"name": "Single Transformer", "use_transformer": True, "use_gru": False, "gcn_type": "none",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    4: {"name": "Trans + Phy-GCN", "use_transformer": True, "use_gru": False, "gcn_type": "physical",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    5: {"name": "Trans + Adp-GCN", "use_transformer": True, "use_gru": False, "gcn_type": "adaptive",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    6: {"name": "Trans + Dual-GCN", "use_transformer": True, "use_gru": False, "gcn_type": "both", "use_concat": False,
        "use_fault_weight": False, "use_carbon_loss": False},
    7: {"name": "Classic (GRU+GCN)", "use_transformer": False, "use_gru": True, "gcn_type": "physical",
        "use_concat": False, "use_fault_weight": False, "use_carbon_loss": False},
    8: {"name": "PASTG w/o Adaptive", "use_transformer": True, "use_gru": False, "gcn_type": "physical",
        "use_concat": True, "use_fault_weight": True, "use_carbon_loss": True},
    9: {"name": "PASTG w/o Concat", "use_transformer": True, "use_gru": False, "gcn_type": "both", "use_concat": False,
        "use_fault_weight": True, "use_carbon_loss": True},
    10: {"name": "PASTG w/o Fault Weight", "use_transformer": True, "use_gru": False, "gcn_type": "both",
         "use_concat": True, "use_fault_weight": False, "use_carbon_loss": True},
    11: {"name": "PASTG w/o Physics Loss", "use_transformer": True, "use_gru": False, "gcn_type": "both",
         "use_concat": True, "use_fault_weight": True, "use_carbon_loss": False},
    12: {"name": "PASTG (Proposed)", "use_transformer": True, "use_gru": False, "gcn_type": "both", "use_concat": True,
         "use_fault_weight": True, "use_carbon_loss": True,
         "pretrained_path": os.path.join(RESULT_DIR, 'best_PASTG_model.pth')},
}


def run_all():
    res_path = os.path.join(RESULT_DIR, 'baseline_results2.json')
    if os.path.exists(res_path):
        print(f">>> 加载已有结果: {res_path} (如需重跑请删除此文件)")
        with open(res_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
        for eid, cfg in EXPERIMENTS.items():
            rmse, mae, r2, res = train_and_eval_model(eid, cfg)
            results[str(eid)] = {'name': cfg['name'], 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2),
                                 'residual': float(res)}
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=4)

    # 绘图逻辑
    def get_data(ids):
        return [results[i]['name'] for i in ids], [results[i]['rmse'] for i in ids], [results[i]['mae'] for i in ids], [
            results[i]['r2'] for i in ids]

    n1, rm1, ma1, r21 = get_data(['1', '2', '3', '4', '5', '6', '7', '12'])
    n2, rm2, ma2, r22 = get_data(['8', '9', '10', '11', '12'])

    plot_metrics(n1, rm1, ma1, r21, 'fig_baseline_comparison2.png', 'Baseline Methods Comparison (RMSE, MAE & R²)')
    plot_metrics(n2, rm2, ma2, r22, 'fig_ablation_study2.png', 'Ablation Study of PASTG Components')

    # 物理残差对比图保持不变
    plt.figure(figsize=(7, 6))
    labels = ['PASTG w/o Physics\n(Data-Only)', 'PASTG\n(Physics-Informed)']
    residuals = [results['11']['residual'], results['12']['residual']]

    bars = plt.bar(labels, residuals, color=['#d62728', '#1f77b4'], width=0.4, alpha=0.9)
    plt.ylabel("Carbon Imbalance Residual (g/kWh)", fontweight='bold', fontsize=12)
    # 增加图表标题
    plt.title("Trade-off Analysis: Physical Consistency Verification", fontweight='bold', fontsize=14)
    plt.grid(axis='y', ls='--', alpha=0.7)

    # 遍历柱子，在顶部添加具体数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(residuals) * 0.015),
                 f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig_physics_residual2.png'), dpi=300)
    plt.close()
    print(">>> 实验全数完成！结果已生成。")


if __name__ == "__main__":
    run_all()