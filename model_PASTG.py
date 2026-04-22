import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 从配置文件获取参数
try:
    from data_loader import CONFIG
except ImportError:
    CONFIG = None

class AdaptiveGraphLayer(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super(AdaptiveGraphLayer, self).__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, node_dim), requires_grad=True)
        self.node_vec2 = nn.Parameter(torch.randn(num_nodes, node_dim), requires_grad=True)

    def forward(self):
        similarity = torch.mm(self.node_vec1, self.node_vec2.transpose(0, 1))
        adp = F.softmax(F.relu(similarity), dim=1)
        return adp

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.linear(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :]

class TS_Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_nodes, dropout=0.1):
        super(TS_Block, self).__init__()
        self.temporal_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)

        self.gcn_static = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_adp = GCNLayer(hidden_dim, hidden_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

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

        attn_out, _ = self.temporal_att(x_t_in, x_t_in, x_t_in)
        x_t_out = self.norm_t(x_t_in + attn_out)

        x = x_t_out.reshape(B, N, T, D)
        x_s_in = x.permute(0, 2, 1, 3).reshape(B * T, N, D)

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

class PASTG(nn.Module):
    def __init__(self, num_nodes, in_dim=5, out_dim=1, out_len=1, hidden_dim=64):
        super(PASTG, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        self.out_dim = out_dim

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.layer1 = TS_Block(hidden_dim, 4, num_nodes)
        self.layer2 = TS_Block(hidden_dim, 4, num_nodes)
        self.adaptive_graph = AdaptiveGraphLayer(num_nodes, 20)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * out_len)
        )

    def forward(self, x_hist, x_now, edge_index):
        B, T, N, F = x_hist.shape
        x = x_hist.permute(0, 2, 1, 3)

        adj = torch.eye(N).to(x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

        d_inv = torch.pow(adj.sum(1), -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        norm_adj = torch.mm(torch.diag(d_inv), torch.mm(adj, torch.diag(d_inv)))

        x = self.pos_encoder(self.embedding(x))
        x = self.layer2(self.layer1(x, norm_adj, self.adaptive_graph()), norm_adj, self.adaptive_graph())

        x_context = x[:, :, -1, :]
        x_combined = torch.cat([x_context, x_now], dim=-1)

        out = self.output_head(x_combined)
        return out.reshape(B, N, self.out_len, self.out_dim).permute(0, 2, 1, 3)

def get_model():
    in_d = 5 # 匹配精准定位特征维度
    if CONFIG is None:
        return PASTG(num_nodes=39, in_dim=in_d, out_dim=1, out_len=1)

    print(f">>> 正在初始化 PASTG (精准定位加权版 | {in_d}维特征输入) ...")
    return PASTG(
        num_nodes=39,
        in_dim=in_d,
        out_dim=1,
        out_len=CONFIG['out_len'],
        hidden_dim=64
    )