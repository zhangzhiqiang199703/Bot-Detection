import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report
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
    test_data  = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test  = test_data.iloc[:, :-1].values
    y_test  = test_data.iloc[:, -1].values

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test  = np.where(np.isinf(X_test),  np.nan, X_test)

    X_train = np.clip(X_train, -1e20, 1e20)
    X_test  = np.clip(X_test,  -1e20, 1e20)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test  = label_encoder.transform(y_test)

    class_names = [str(cls) for cls in label_encoder.classes_]
    return X_train, y_train, X_test, y_test, scaler, class_names



class FWAM(nn.Module):
    """多尺度频域小波门控注意力 (Multi-Scale Wavelet-Gated Attention)"""
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        pad       = (4 - dim % 4) % 4
        self.pad  = pad
        dim_p     = dim + pad
        self.dim_p = dim_p
        q         = dim_p // 4         

        self.gate_low = nn.Sequential(
            nn.LayerNorm(q),
            nn.Linear(q, q), nn.GELU(),
            nn.Linear(q, q), nn.Sigmoid()
        )
        self.gate_high1 = nn.Sequential(
            nn.LayerNorm(dim_p // 2),
            nn.Linear(dim_p // 2, dim_p // 2), nn.GELU(),
            nn.Linear(dim_p // 2, dim_p // 2), nn.Sigmoid()
        )
        self.gate_high2 = nn.Sequential(
            nn.LayerNorm(q),
            nn.Linear(q, q), nn.GELU(),
            nn.Linear(q, q), nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU()
        )

    def _haar(self, x):
        e, o = x[:, 0::2], x[:, 1::2]
        return (e + o) * 0.5, (e - o) * 0.5

    def _ihaar(self, low, high):
        e = (low + high) * 0.5
        o = (low - high) * 0.5
        B, L = e.shape
        out  = torch.stack([e, o], dim=2).reshape(B, L * 2)
        return out

    def forward(self, x):
        identity = x
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))

        low1, high1 = self._haar(x)
        high1 = high1 * (1.0 + self.gate_high1(high1))

        low2, high2 = self._haar(low1)
        low2  = low2  * self.gate_low(low2)
        high2 = high2 * (1.0 + self.gate_high2(high2))

        low1_r = self._ihaar(low2, high2)
        x_r    = self._ihaar(low1_r, high1)

        if self.pad > 0:
            x_r = x_r[:, :self.dim]

        return self.mix(x_r) + identity


class DHIM(nn.Module):
    """动态特征超图交互 (Dynamic Hypergraph Feature Interaction)"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads   
        self.scale     = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.pos_embed = nn.Parameter(torch.empty(1, dim, self.head_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02, a=-0.04, b=0.04)

        self.q_proj   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, D  = x.shape
        H     = self.num_heads
        Hd    = self.head_dim

        x_val  = x.unsqueeze(-1)                       
        tokens = x_val * self.pos_embed + self.pos_embed  

        tokens_n = F.layer_norm(tokens, [Hd])

        q = self.q_proj(tokens_n) 
        k = self.k_proj(tokens_n)
        v = self.v_proj(tokens_n)

        head_size = Hd // H
        q = q.view(B, D, H, head_size).permute(0, 2, 1, 3)  
        k = k.view(B, D, H, head_size).permute(0, 2, 1, 3)
        v = v.view(B, D, H, head_size).permute(0, 2, 1, 3)

        scale = head_size ** -0.5
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn  = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, D, Hd)

        feat = (out * self.pos_embed).sum(dim=-1)

        feat  = self.norm1(x + self.out_proj(feat))
        feat  = self.norm2(feat + self.ffn(feat))
        return feat


class AdvancedTeacherIDS(nn.Module):
    def __init__(self, input_dim, num_classes, use_fwam=True, use_dhim=True):
        super().__init__()
        hidden_dim = 256
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.fwam = FWAM(hidden_dim) if use_fwam else nn.Identity()
        self.dhim = DHIM(hidden_dim, num_heads=4) if use_dhim else nn.Identity()
        self.fc   = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x      = self.in_proj(x)
        x      = self.fwam(x)
        x      = self.dhim(x)
        feat   = self.fc(x)
        logits = self.classifier(feat)
        return logits, feat


def evaluate_model(model, test_loader, device, y_test_true, class_names, model_label):
    model.eval()
    preds_all = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            out     = model(X_batch)
            preds   = torch.argmax(out[0], dim=1)
            preds_all.extend(preds.cpu().numpy())

    preds_all = np.array(preds_all)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test_true, preds_all, average='weighted', zero_division=0
    )

    print(f"分类报告")
    print(classification_report(
        y_test_true, preds_all,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))
    return precision, recall, fscore



def train_model(model, train_loader, device, epochs=100, lr=0.001):
    optimizer   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion   = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out    = model(X_batch)
            loss   = criterion(out[0], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()


def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "../train.csv"
    test_path  = "../test.csv"
    X_train, y_train, X_test, y_test, scaler, class_names = load_enhanced_data(
        train_path, test_path
    )

    input_dim   = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.LongTensor(y_test)

    def make_loaders(seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        tl = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=128, shuffle=True, generator=g
        )
        vl = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=128, shuffle=False
        )
        return tl, vl

    set_seeds(42)
    train_loader, test_loader = make_loaders(42)
    full_model = AdvancedTeacherIDS(input_dim, num_classes, use_fwam=True, use_dhim=True)
    train_model(full_model, train_loader, device, epochs=100, lr=0.001)
    p_full, r_full, f_full = evaluate_model(full_model, test_loader, device, y_test,
                                            class_names, "Full Teacher (Ours)")
    torch.save(full_model.state_dict(), "full_teacher.pth")

if __name__ == "__main__":
    main()