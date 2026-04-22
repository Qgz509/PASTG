import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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

EPOCHS = 50  # 基线模型训练轮数
PATIENCE = 15  # 早停耐心值
GAMMA_VAL = 500.0  # 物理守恒极大化权重
FAULT_WEIGHT = 5.0  # N-1 故障节点的惩罚倍数

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 物理信息与故障加权引导的损失函数 (纯 MSE 核心)
# ==========================================
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, scaler, gamma, fault_weight, use_fault_weight):
        super(PhysicsGuidedLoss, self).__init__()
        # 使用 reduction='none' 以便后续应用节点级加权
        self.mse_loss = nn.MSELoss(reduction='none')
        self.gamma = gamma
        self.scaler = scaler
        self.fault_weight = fault_weight
        self.use_fault_weight = use_fault_weight  # 故障加权开关

    def forward(self, pred, target, pb_matrix, c_gen, p_gen):
        # pred: (B, 1, N, 1) -> 碳浓度
        # target: (B, 1, N, 2) -> [碳浓度, 故障标签]
        pred_c = pred[:, 0, :, 0]
        target_c = target[:, 0, :, 0]

        # 计算基础 MSE
        base_mse = self.mse_loss(pred_c, target_c)

        # --- 核心消融：故障加权 MSE vs 标准等权 MSE ---
        if self.use_fault_weight:
            target_fault = target[:, 0, :, 1]  # 提取精准定位标签
            # 故障节点放大惩罚 (1 + 5 = 6倍)
            weight_mask = 1.0 + target_fault * self.fault_weight
            loss_data = torch.mean(base_mse * weight_mask)
        else:
            # 退化为标准的均等 MSE 损失
            loss_data = torch.mean(base_mse)

        # 如果不使用碳守恒约束，直接返回数据损失
        if self.gamma <= 0:
            return loss_data, loss_data.item(), 0.0

        # --- 碳流物理守恒约束 (PINN) ---
        c_mean = torch.tensor(self.scaler.mean_[3::4], device=pred.device).float()
        c_scale = torch.tensor(self.scaler.scale_[3::4], device=pred.device).float()

        pred_c_real = torch.relu(pred_c * c_scale + c_mean)
        pb_pu, c_gen_pu, p_gen_pu = pb_matrix / 100.0, c_gen / 100.0, p_gen / 100.0

        p_sum = pb_pu.sum(dim=2) + p_gen_pu
        p_sum[p_sum < 1e-5] = 1.0

        output_carbon = p_sum * pred_c_real
        input_carbon = c_gen_pu + torch.bmm(pb_pu, pred_c_real.unsqueeze(-1)).squeeze(-1)

        # 物理守恒残差的 L2 惩罚
        loss_cons = torch.mean(((output_carbon - input_carbon) / 850.0) ** 2)

        return loss_data + self.gamma * loss_cons, loss_data.item(), loss_cons.item()


def calculate_physics_residual(model, test_loader, scaler, edge_index):
    model.eval()
    total_residual, total_samples = 0.0, 0
    c_mean = torch.tensor(scaler.mean_[3::4], device=DEVICE).float()
    c_scale = torch.tensor(scaler.scale_[3::4], device=DEVICE).float()

    with torch.no_grad():
        for batch in test_loader:
            x_hist, x_now, _, pb_now, c_gen_now, p_gen_now = [b.to(DEVICE) for b in batch]
            pred = model(x_hist, x_now, edge_index)
            pred_c_real = torch.relu(pred[:, 0, :, 0] * c_scale + c_mean)

            pb_pu, c_gen_pu, p_gen_pu = pb_now / 100.0, c_gen_now / 100.0, p_gen_now / 100.0
            p_sum = pb_pu.sum(dim=2) + p_gen_pu
            p_sum[p_sum < 1e-5] = 1.0

            output_carbon = p_sum * pred_c_real
            input_carbon = c_gen_pu + torch.bmm(pb_pu, pred_c_real.unsqueeze(-1)).squeeze(-1)

            residual = torch.abs(output_carbon - input_carbon)
            total_residual += residual.sum().item()
            total_samples += residual.numel()

    return total_residual / total_samples


# ==========================================
# 2. 灵活可配置的模型架构 (集成特征级动态门控)
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
        if self.gcn_type in ['physical', 'both']:
            self.gcn_static = GCNLayer(hidden_dim, hidden_dim)
        if self.gcn_type in ['adaptive', 'both']:
            self.gcn_adp = GCNLayer(hidden_dim, hidden_dim)

        if self.gcn_type == 'both':
            self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.norm_gcn = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
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
            last_frame = x_t_in[:, -1:, :]
            x_t_out = last_frame.expand(-1, T, -1)

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
                out_static = self.gcn_static(x_s_in, adj_static)
                out_adp = self.gcn_adp(x_s_in, adj_adp)
                gate = self.fusion_gate(torch.cat([out_static, out_adp], dim=-1))
                x_gcn_out = gate * out_static + (1.0 - gate) * out_adp

            x_gcn_out = F.relu(x_gcn_out)
            x_gcn_out = self.dropout(x_gcn_out)
            x_s_out = self.norm_gcn(x_s_in + x_gcn_out)

        x = x_s_out.reshape(B, T, N, D).permute(0, 2, 1, 3)
        ff_out = self.feed_forward(x)
        x = self.norm_ff(x + ff_out)

        return x


class FlexiblePASTG(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, out_len, config, hidden_dim=64):
        super(FlexiblePASTG, self).__init__()
        self.num_nodes, self.out_len, self.out_dim, self.config = num_nodes, out_len, out_dim, config

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.layer1 = Flexible_TS_Block(hidden_dim, 4, num_nodes, config)
        self.layer2 = Flexible_TS_Block(hidden_dim, 4, num_nodes, config)

        if config['gcn_type'] in ['adaptive', 'both']:
            self.adaptive_graph = AdaptiveGraphLayer(num_nodes, 20)

        # 实时预测模型：如果开启了拼接，则多接入 3 维实时物理信息 x_now
        head_in = hidden_dim + 3 if config['use_concat'] else hidden_dim
        self.output_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * out_len)  # 强制单通道输出 (仅碳强度)
        )

    def forward(self, x_hist, x_now, edge_index):
        B, T, N, F = x_hist.shape
        x = x_hist.permute(0, 2, 1, 3)

        adj_static = torch.eye(N).to(x.device)
        adj_static[edge_index[0], edge_index[1]] = 1.0
        adj_static[edge_index[1], edge_index[0]] = 1.0

        d_inv = torch.pow(adj_static.sum(1), -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        norm_adj = torch.mm(torch.diag(d_inv), torch.mm(adj_static, torch.diag(d_inv)))

        adj_adp = self.adaptive_graph() if self.config['gcn_type'] in ['adaptive', 'both'] else None

        x = self.pos_encoder(self.embedding(x))
        x = self.layer2(self.layer1(x, norm_adj, adj_adp), norm_adj, adj_adp)
        x_context = x[:, :, -1, :]

        x_combined = torch.cat([x_context, x_now], dim=-1) if self.config['use_concat'] else x_context
        out = self.output_head(x_combined)
        return out.reshape(B, N, self.out_len, self.out_dim).permute(0, 2, 1, 3)

class PureTransformer_Baseline(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, out_len, hidden_dim=64):
        super(PureTransformer_Baseline, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        self.out_dim = out_dim

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * out_len)
        )

    def forward(self, x_hist, x_now, edge_index):
        B, T, N, F = x_hist.shape

        x = x_hist.permute(0, 2, 1, 3)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # 加完位置编码后，再将 Node 压平到 Batch 维度 (B*N, T, hidden) 送入标准 Transformer
        x = x.reshape(B * N, T, -1)

        # 纯时序自注意力提取
        x = self.transformer_encoder(x)

        # 截取最后一个时间步的隐状态
        x_context = x[:, -1, :]  # 形状: (B*N, hidden_dim)

        # 直接输出预测 (无物理拼接)
        out = self.output_head(x_context)

        # 把合并的维度重新拆解开来，恢复标准输出格式
        return out.reshape(B, N, self.out_len, self.out_dim).permute(0, 2, 1, 3)




# ==========================================
# 3. 实验配置表
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

    # 核心消融实验部分 (控制变量)
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
         "pretrained_path": os.path.join(RESULT_DIR, 'best_PASTG_model.pth')},  # 若跑过 train_PASTG.py，直接加载权重
}


# ==========================================
# 4. 统一训练与智能评估包装器
# ==========================================
def train_and_eval_model(exp_id, config):
    print(f"\n[{exp_id}/12] 正在处理: {config['name']} ...")

    train_loader, val_loader, test_loader, edge_index, scaler = load_data()
    edge_index = edge_index.to(DEVICE)

    # 模型输入设定为 5 维 (P, Q, V, C, Fault)
    if exp_id == 3:
        model = PureTransformer_Baseline(num_nodes=39, in_dim=5, out_dim=1, out_len=CONFIG['out_len']).to(DEVICE)
    else:
        model = FlexiblePASTG(num_nodes=39, in_dim=5, out_dim=1, out_len=CONFIG['out_len'], config=config).to(DEVICE)

    if 'pretrained_path' in config and os.path.exists(config['pretrained_path']):
        print(f"    [!] 检测到预训练权重，直接加载以节省时间: {config['pretrained_path']}")
        model.load_state_dict(torch.load(config['pretrained_path'], map_location=DEVICE), strict=False)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        gamma = GAMMA_VAL if config['use_carbon_loss'] else 0.0

        # 传入 MSE 的故障权重开关
        criterion = PhysicsGuidedLoss(
            scaler=scaler,
            gamma=gamma,
            fault_weight=FAULT_WEIGHT,
            use_fault_weight=config['use_fault_weight']
        )

        best_val_loss = float('inf')
        best_weights = None
        patience_cnt = 0

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                x_hist, x_now, y, pb_now, c_gen_now, p_gen_now = [b.to(DEVICE) for b in batch]

                optimizer.zero_grad()
                pred = model(x_hist, x_now, edge_index)
                loss, _, _ = criterion(pred, y, pb_now, c_gen_now, p_gen_now)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            model.eval()
            val_ls = []
            with torch.no_grad():
                for batch in val_loader:
                    x_hist, x_now, y, pb_now, c_gen_now, p_gen_now = [b.to(DEVICE) for b in batch]
                    pred = model(x_hist, x_now, edge_index)
                    _, l_data, _ = criterion(pred, y, pb_now, c_gen_now, p_gen_now)
                    val_ls.append(l_data)

            avg_val = np.mean(val_ls)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_weights = model.state_dict().copy()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break

        if best_weights is not None:
            model.load_state_dict(best_weights)

    # --- 统一测试评估 ---
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_hist, x_now, y, _, _, _ = [b.to(DEVICE) for b in batch]
            all_p.append(model(x_hist, x_now, edge_index))
            all_y.append(y)

    p_np = torch.cat(all_p).cpu().numpy()
    t_np = torch.cat(all_y).cpu().numpy()

    c_scale = scaler.scale_[3::4].reshape(1, -1)
    c_mean = scaler.mean_[3::4].reshape(1, -1)

    # 反归一化 (仅提取碳浓度，丢掉标签里的第 5 维)
    c_pred = np.maximum(p_np[:, 0, :, 0] * c_scale + c_mean, 0)
    c_true = t_np[:, 0, :, 0] * c_scale + c_mean

    y_t, y_p = c_true.flatten(), c_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    residual = calculate_physics_residual(model, test_loader, scaler, edge_index)

    print(f"    [✔] 评估完成 -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Residual: {residual:.4f}")
    return rmse, r2, residual


# ==========================================
# 5. 学术绘图与其他
# ==========================================
def plot_metrics(names, rmses, r2s, filename, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, rmses, width, label='RMSE', color='#1f77b4', alpha=0.8)
    ax1.set_ylabel('RMSE (g/kWh)', color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=10)

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#1f77b4')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, r2s, width, label='R² Score', color='#ff7f0e', alpha=0.8)
    ax2.set_ylabel('R² Score', color='#ff7f0e', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim([max(0, min(r2s) - 0.1), min(1.0, max(r2s) + 0.05)])

    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#ff7f0e')

    fig.suptitle(title, fontweight='bold', fontsize=15)
    fig.tight_layout()

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    save_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_residuals(res_no_phy, res_pastg, filename):
    labels = ['PASTG w/o Physics Loss\n(Data-Driven Only)', 'PASTG (Proposed)\n(Physics-Informed)']
    residuals = [res_no_phy, res_pastg]

    fig = plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, residuals, color=['#d62728', '#1f77b4'], alpha=0.85, width=0.45)
    plt.ylabel("Average Carbon Imbalance Residual (g/kWh)", fontweight='bold', fontsize=12)
    plt.title("Trade-off Analysis: Physical Consistency Verification", fontweight='bold', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(residuals) * 0.015),
                 f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    save_path = os.path.join(RESULT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def run_all():
    results_file = os.path.join(RESULT_DIR, 'baseline_results.json')

    if os.path.exists(results_file):
        print(f">>> 读取本地已有实验结果 {results_file}...")
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
        for exp_id, config in EXPERIMENTS.items():
            rmse, r2, residual = train_and_eval_model(exp_id, config)
            results[str(exp_id)] = {'name': config['name'], 'rmse': float(rmse), 'r2': float(r2),
                                    'residual': float(residual)}

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    # 拆分两张图：Group 1 画前 7 个传统基线 + 第 12 个最终模型
    group1_ids = ['1', '2', '3', '4', '5', '6', '7', '12']
    g1_names = [results[i]['name'] for i in group1_ids]
    g1_rmse = [results[i]['rmse'] for i in group1_ids]
    g1_r2 = [results[i]['r2'] for i in group1_ids]

    # Group 2 画 8 到 12 这五个内部组件消融模型
    group2_ids = ['8', '9', '10', '11', '12']
    g2_names = [results[i]['name'] for i in group2_ids]
    g2_rmse = [results[i]['rmse'] for i in group2_ids]
    g2_r2 = [results[i]['r2'] for i in group2_ids]

    # 提取物理守恒验证的残差对比 (Exp 11 vs Exp 12)
    res_no_phy = results['11']['residual']
    res_pastg = results['12']['residual']

    print("\n>>> 开始绘制学术对比图表...")
    plot_metrics(g1_names, g1_rmse, g1_r2, 'fig_baseline_comparison.png', 'Baseline Methods Comparison (RMSE & R²)')
    plot_metrics(g2_names, g2_rmse, g2_r2, 'fig_ablation_study.png', 'Ablation Study of PASTG Components')
    plot_residuals(res_no_phy, res_pastg, 'fig_physics_residual.png')
    print(">>> 实验评估流程全数完成！核心图表已生成在 'result/' 目录下。")


if __name__ == "__main__":
    run_all()