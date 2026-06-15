import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)

set_seeds(42)   


def load_enhanced_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, y_train, X_test, y_test, [str(c) for c in le.classes_]


class FWAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        pad = (4 - dim % 4) % 4
        self.pad = pad
        dim_p = dim + pad
        q = dim_p // 4
        self.gate_low = nn.Sequential(
            nn.LayerNorm(q), nn.Linear(q, q), nn.GELU(), nn.Linear(q, q), nn.Sigmoid()
        )
        self.gate_high1 = nn.Sequential(
            nn.LayerNorm(dim_p // 2), nn.Linear(dim_p // 2, dim_p // 2), nn.GELU(), nn.Linear(dim_p // 2, dim_p // 2), nn.Sigmoid()
        )
        self.gate_high2 = nn.Sequential(
            nn.LayerNorm(q), nn.Linear(q, q), nn.GELU(), nn.Linear(q, q), nn.Sigmoid()
        )
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())

    def _haar(self, x):
        e, o = x[:, 0::2], x[:, 1::2]
        return (e + o) * 0.70710678, (e - o) * 0.70710678

    def _ihaar(self, low, high):
        e = (low + high) * 0.70710678
        o = (low - high) * 0.70710678
        B, L = e.shape
        out = torch.stack([e, o], dim=2).reshape(B, L * 2)
        return out

    def forward(self, x):
        identity = x
        if self.pad > 0: x = F.pad(x, (0, self.pad))
        low1, high1 = self._haar(x)
        high1 = high1 * (1.0 + self.gate_high1(high1))
        low2, high2 = self._haar(low1)
        low2 = low2 * self.gate_low(low2)
        high2 = high2 * (1.0 + self.gate_high2(high2))
        low1_r = self._ihaar(low2, high2)
        x_r = self._ihaar(low1_r, high1)
        if self.pad > 0: x_r = x_r[:, :self.dim]
        return self.mix(x_r) + identity

class DHIM(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(torch.empty(1, dim, self.head_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))

    def forward(self, x):
        B, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        x_val = x.unsqueeze(-1)
        tokens = x_val * self.pos_embed + self.pos_embed
        tokens_n = F.layer_norm(tokens, [Hd])
        q = self.q_proj(tokens_n).view(B, D, H, Hd // H).permute(0, 2, 1, 3)
        k = self.k_proj(tokens_n).view(B, D, H, Hd // H).permute(0, 2, 1, 3)
        v = self.v_proj(tokens_n).view(B, D, H, Hd // H).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * ((Hd // H) ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, D, Hd)
        feat = (out * self.pos_embed).sum(dim=-1)
        feat = self.norm1(x + self.out_proj(feat))
        return self.norm2(feat + self.ffn(feat))

class AdvancedTeacherIDS(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        hidden_dim = 256
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.fwam = FWAM(hidden_dim)
        self.dhim = DHIM(hidden_dim, num_heads=4)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.fwam(x)
        x = self.dhim(x)
        feat = self.fc(x)
        logits = self.classifier(feat)
        return logits, feat


# 轻量化学生模型（保留 FWAM 和 DHIM 模块）
class StudentIDS(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        # 输入投影层：将原始特征映射到 hidden_dim
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 保留 FWAM 和 DHIM，但使用更小的 hidden_dim
        self.fwam = FWAM(hidden_dim)
        self.dhim = DHIM(hidden_dim, num_heads=4) 
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.fwam(x)
        x = self.dhim(x)
        feat = self.fc(x)
        logits = self.classifier(feat)
        return logits, feat

# 蒸馏损失函数 (DSTMD Loss)
class DSTMDLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  

    def forward(self, s_logits, t_logits, s_feat, t_feat, target):
        with torch.no_grad():
            t_probs = F.softmax(t_logits, dim=1)
            entropy = -torch.sum(t_probs * torch.log(t_probs + 1e-6), dim=1)
            T = 1.0 + entropy.mean() # 熵越高，温度越高，平滑软标签

        # 软标签蒸馏
        loss_soft = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction='batchmean'
        ) * (T**2)

        # 拓扑流形一致性损失 (Manifold Consistency)
        s_rel = torch.matmul(F.normalize(s_feat, p=2, dim=1), F.normalize(s_feat, p=2, dim=1).t())
        t_rel = torch.matmul(F.normalize(t_feat, p=2, dim=1), F.normalize(t_feat, p=2, dim=1).t())
        loss_topo = F.mse_loss(s_rel, t_rel)

        # 交叉熵损失
        loss_ce = F.cross_entropy(s_logits, target)

        return (1 - self.alpha) * loss_ce + self.alpha * loss_soft + self.beta * loss_topo


def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, class_names = load_enhanced_data("../train.csv", "../test.csv")
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=128, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(class_names)
    teacher = AdvancedTeacherIDS(input_dim, num_classes).to(device)
    
    teacher_pth = "full_teacher.pth"
    if os.path.exists(teacher_pth):
        teacher.load_state_dict(torch.load(teacher_pth, map_location=device))
    else:
        print(f"错误: 找不到 {teacher_pth}，请先运行教师训练代码！")
        return

    # 统计教师参数
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"教师模型参数量: {t_params:,}")

    student = StudentIDS(input_dim, num_classes, hidden_dim=128).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"学生模型参数量: {s_params:,} (压缩率: {t_params/s_params:.2f}x)")

    # 蒸馏训练
    optimizer = optim.AdamW(student.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = DSTMDLoss(alpha=0.5, beta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    teacher.eval()
    for epoch in range(300):
        student.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                t_logits, t_feat = teacher(X_batch)
            
            s_logits, s_feat = student(X_batch)
            loss = criterion(s_logits, t_logits, s_feat, t_feat, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f}")

    # 最终评估
    def evaluate(model, loader, label):
        model.eval()
        preds_all = []
        with torch.no_grad():
            for X, _ in loader:
                out, _ = model(X.to(device))
                preds_all.extend(torch.argmax(out, dim=1).cpu().numpy())
        print(f"\n--- {label} 评估报告 ---")
        print(classification_report(y_test, preds_all, target_names=class_names, digits=4))
        return precision_recall_fscore_support(y_test, preds_all, average='weighted')[2]

    t_f1 = evaluate(teacher, test_loader, "教师模型 (Full)")
    s_f1 = evaluate(student, test_loader, "蒸馏学生模型 (DSTMD)")

    student_save_path = "distilled_student.pth"
    torch.save(student.state_dict(), student_save_path)

    print(f"教师 F1: {t_f1:.4f} | 学生 F1: {s_f1:.4f}")
    print(f"性能保持率: {(s_f1/t_f1)*100:.2f}%")
    print(f"参数量缩减: {t_params/s_params:.1f} 倍")

if __name__ == "__main__":
    main()