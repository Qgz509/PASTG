import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import os

from data_loader import load_data, CONFIG
from model_PASTG import get_model

# 配置
RESULT_DIR = 'result'
# 注意加载的是我们训练出的 5维精准定位模型权重
MODEL_PATH = os.path.join(RESULT_DIR, 'best_PASTG_model.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'


# ==========================================
# 1. 辅助函数：反归一化 & 提取故障标签
# ==========================================
def inverse_transform(pred_tensor, true_tensor, scaler):
    p_np = pred_tensor.cpu().numpy()
    t_np = true_tensor.cpu().numpy()

    # 提取 StandardScaler 属性
    c_mean = scaler.mean_[3::4].reshape(1, -1)
    c_scale = scaler.scale_[3::4].reshape(1, -1)

    # 预测输出只有碳浓度 [B, 1(out_len), N, 1(碳)]
    p_inv_c = p_np[:, 0, :, 0] * c_scale + c_mean

    # 真实标签包含碳浓度和故障 [B, 1(out_len), N, 2(碳+Fault)]
    t_inv_c = t_np[:, 0, :, 0] * c_scale + c_mean
    fault_labels = t_np[:, 0, :, 1]  # 提取精准故障标签 (0.0 或 1.0)

    return p_inv_c, t_inv_c, fault_labels


# ==========================================
# 2. 核心评估与多维度绘图逻辑
# ==========================================
def evaluate_best_model():
    print(f"\n>>>启动最优模型独立评估 (Device: {DEVICE})")

    if not os.path.exists(MODEL_PATH):
        print(f"[!] 错误：未找到权重文件 {MODEL_PATH}")
        return

    # 1. 加载数据
    _, _, test_loader, edge_index, scaler = load_data()
    edge_index = edge_index.to(DEVICE)

    # 2. 加载模型 (自动匹配 5 维输入)
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("    [✔] 成功加载最优权重: best_PASTG_model.pth")

    # 3. 全量推理 (使用安全 batch 解包应对各种返回长度)
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_hist, x_now, y = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            pred = model(x_hist, x_now, edge_index)
            all_p.append(pred)
            all_y.append(y)

    # 4. 反归一化并获取故障标签
    c_pred, c_true, fault_labels = inverse_transform(torch.cat(all_p), torch.cat(all_y), scaler)
    c_pred = np.maximum(c_pred, 0)  # 物理非负硬约束

    # ---------------------------------------------------------
    # 图 1 动态追踪对比 (带故障高亮显示)
    # ---------------------------------------------------------
    print("    正在生成 图1: 动态追踪对比 (带 N-1 故障高亮)...")
    time_steps = 168
    nodes_to_plot = [2, 5, 29, 37]
    custom_ticks = list(range(0, 151, 25)) + [time_steps]

    fig1 = plt.figure(figsize=(16, 12))
    for i, n_idx in enumerate(nodes_to_plot):
        ax = plt.subplot(4, 1, i + 1)
        ax.plot(c_true[:time_steps, n_idx], color='dimgray', linestyle='-', linewidth=2.5, label='Ground Truth')
        ax.plot(c_pred[:time_steps, n_idx], color='royalblue', linestyle='--', linewidth=2, label='PASTG')

        # 【学术亮点】：高亮该节点发生故障的时刻
        fault_times = np.where(fault_labels[:time_steps, n_idx] == 1.0)[0]
        for ft in fault_times:
            ax.axvspan(ft - 0.5, ft + 0.5, color='red', alpha=0.3, lw=0)

        # 添加故障图例 (仅添加一次避免重复)
        if len(fault_times) > 0:
            ax.plot([], [], color='red', alpha=0.3, linewidth=8, label='N-1 Fault Impact')

        ax.set_title(f"Carbon Intensity Dynamic Tracking: Bus {n_idx + 1}", fontweight='bold')
        ax.set_ylabel("Carbon (g/kWh)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_xticks(custom_ticks)
        if i == len(nodes_to_plot) - 1:
            ax.set_xlabel("Time Steps (Hours)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig1_dynamic_tracking.png'), dpi=300)

    # ---------------------------------------------------------
    # 图 2 全网快照柱状图
    # ---------------------------------------------------------
    print("    正在生成 图2: 全网节点快照对比...")
    # 智能寻找一个发生过故障的典型时刻
    fault_indices = np.where(fault_labels == 1.0)[0]
    sample_idx = fault_indices[0] if len(fault_indices) > 0 else 10

    bus_ids = np.arange(1, 40)
    fig2 = plt.figure(figsize=(16, 6))

    # 标记出在该时刻发生故障的具体节点
    colors = ['#d62728' if fault_labels[sample_idx, j] == 1.0 else 'dimgray' for j in range(39)]

    plt.bar(bus_ids - 0.2, c_true[sample_idx], width=0.4, label='Ground Truth (Red=Fault Node)', color=colors,
            alpha=0.7)
    plt.bar(bus_ids + 0.2, c_pred[sample_idx], width=0.4, label='PASTG Predicted', color='#1f77b4')

    plt.title(f"Snapshot Prediction for All 39 Nodes (Sample #{sample_idx})", fontweight='bold', fontsize=16)
    plt.xlabel("Bus ID", fontsize=14)
    plt.ylabel("Carbon Intensity (g/kWh)", fontsize=14)
    plt.xticks(bus_ids, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig2_snapshot_bar.png'), dpi=300)

    # ---------------------------------------------------------
    # 图 3 散点图 + 拟合线
    # ---------------------------------------------------------
    print("    正在生成 图3: 散点回归对齐图与拟合线...")
    y_t_flat = c_true.flatten()
    y_p_flat = c_pred.flatten()

    fig3 = plt.figure(figsize=(8, 8))
    plt.scatter(y_t_flat, y_p_flat, alpha=0.1, s=20, color='#1f77b4', label='Data Points', edgecolors='none')

    max_val = max(np.max(y_t_flat), np.max(y_p_flat))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Alignment (y=x)')

    k, b = np.polyfit(y_t_flat, y_p_flat, 1)
    plt.plot(y_t_flat, k * y_t_flat + b, color='#ff7f0e', linestyle='-', linewidth=2, label=f'Fitting Line (k={k:.3f})')

    plt.title(f"Prediction vs Ground Truth (R² = {r2_score(y_t_flat, y_p_flat):.4f})", fontweight='bold')
    plt.xlabel("Ground Truth Carbon Intensity (g/kWh)")
    plt.ylabel("Predicted Carbon Intensity (g/kWh)")
    plt.xlim(-max_val * 0.02, max_val * 1.05)
    plt.ylim(-max_val * 0.02, max_val * 1.05)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig3_regression_fitting.png'), dpi=300)

    # 5. 打印评估结果
    print("\n" + "=" * 50)
    print(f"全局测试集表现:")
    print(f"  ► RMSE: {np.sqrt(mean_squared_error(y_t_flat, y_p_flat)):.4f}")
    print(f"  ► R²:   {r2_score(y_t_flat, y_p_flat):.4f}")
    print(f"  ► 拟合斜率 k: {k:.4f}")
    print("=" * 50)
    print("结果图已更新至 'result/' 文件夹。")


if __name__ == "__main__":
    evaluate_best_model()