import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import joblib
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedFeatureEngineer:
    def __init__(self):
        self.session_timeout = 1800 
        
    def extract_temporal_features(self, df):
        temporal_features =[]
        prev_timestamp_map = {}  
        session_id_map = {}     

        for idx, row in df.iterrows():
            ip = row['IP']
            try:
                timestamp = row['TimeStamp'] 
                
                hour = timestamp.hour
                minute = timestamp.minute
                day_of_week = timestamp.dayofweek
                day_of_month = timestamp.day
                month = timestamp.month
                
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                day_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_cos = np.cos(2 * np.pi * day_of_week / 7)
                
                time_diff = 0
                is_new_session = True
                if ip in prev_timestamp_map:
                    time_diff = (timestamp - prev_timestamp_map[ip]).total_seconds()
                    if time_diff <= self.session_timeout:
                        is_new_session = False
                    
                if is_new_session:
                    session_id_map[ip] = session_id_map.get(ip, 0) + 1
                
                is_peak_hours = int(9 <= hour <= 18)
                is_late_night = int(0 <= hour <= 6)
                is_weekend = int(day_of_week >= 5)
                is_rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
                
                temporal_vector =[
                    hour_sin, hour_cos, day_sin, day_cos,
                    time_diff, int(is_new_session), session_id_map.get(ip, 0),
                    is_peak_hours, is_late_night, is_weekend, is_rush_hour,
                    minute, day_of_month, month
                ]
                temporal_features.append(temporal_vector)
                prev_timestamp_map[ip] = timestamp
                
            except Exception as e:
                temporal_features.append([0] * 14)
        
        return np.array(temporal_features)
    
    def extract_behavioral_features(self, df, ip_stats_map=None):
        behavioral_features =[]
        global_avg = None
        if ip_stats_map and len(ip_stats_map) > 0:
            all_features = list(ip_stats_map.values())
            global_avg = np.mean(all_features, axis=0)
        
        for ip in df['IP']:
            if ip_stats_map and ip in ip_stats_map:
                behavioral_features.append(ip_stats_map[ip])
            else:
                if global_avg is not None:
                    behavioral_features.append(global_avg.tolist())
                else:
                    behavioral_features.append([0] * 7)
        
        return np.array(behavioral_features)
    
    def compute_ip_stats_for_training(self, df):
        ip_stats = df.groupby('IP').agg(
            request_count=('URI', 'count'),
            method_dist=('RequestMethod', lambda x: x.value_counts().to_dict()),
            status_dist=('StatusCode', lambda x: x.value_counts().to_dict()),
            first_seen=('TimeStamp', 'min'),
            last_seen=('TimeStamp', 'max')
        ).reset_index()
        
        ip_feature_map = {}
        for _, row in ip_stats.iterrows():
            features =[]
            features.append(row['request_count'])
            
            method_dist = row['method_dist']
            total_requests = sum(method_dist.values())
            get_ratio = method_dist.get('GET', 0) / (total_requests + 1e-6)
            post_ratio = method_dist.get('POST', 0) / (total_requests + 1e-6)
            features.extend([get_ratio, post_ratio])
            
            status_dist = row['status_dist']
            status_2xx = sum(v for k, v in status_dist.items() if 200 <= k < 300) / (total_requests + 1e-6)
            status_4xx = sum(v for k, v in status_dist.items() if 400 <= k < 500) / (total_requests + 1e-6)
            status_5xx = sum(v for k, v in status_dist.items() if 500 <= k < 600) / (total_requests + 1e-6)
            features.extend([status_2xx, status_4xx, status_5xx])
            
            duration = (row['last_seen'] - row['first_seen']).total_seconds() / 3600
            features.append(duration)
            
            ip_feature_map[row['IP']] = features
        
        return ip_feature_map

def safe_int_convert(value, default=0):
    try:
        if pd.isna(value) or value == '' or value == '-': return default
        return int(float(str(value)))
    except: return default

def calculate_entropy(text):
    if not text: return 0
    text = str(text)
    counter = Counter(text)
    text_len = len(text)
    entropy = 0
    for count in counter.values():
        p = count / text_len
        entropy -= p * np.log2(p)
    return entropy

def extract_handcrafted_features(df):
    features =[]
    for _, row in df.iterrows():
        ua = str(row['UserAgent']).lower()
        uri = str(row['URI']).lower()
        
        ua_has_suspicious = int(any(p in ua for p in ['bot', 'crawler', 'spider', 'scraper', 'python', 'java']))
        uri_has_suspicious = int(any(p in uri for p in['admin', 'login', 'config', 'phpmyadmin', 'wp-admin']))            
        is_https = int('https' in str(row['Protocol']).lower())
        status_code = safe_int_convert(row['StatusCode'], 200)
        bytes_val = safe_int_convert(row['Bytes'])
        bytes_log = np.log1p(max(bytes_val, 0))
        ref_len = len(str(row['Referrer']))
        
        features.append([
            len(ua), len(uri), ref_len,
            calculate_entropy(ua), calculate_entropy(uri),
            ua_has_suspicious, uri_has_suspicious, is_https,
            bytes_log, status_code
        ])
    return np.array(features)

class HybridTextProcessor:
    def __init__(self, max_features=2000):
        self.tfidf_ua = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=(2, 5))
        self.tfidf_uri = TfidfVectorizer(max_features=1500, analyzer='char', ngram_range=(2, 4))
        self.tfidf_ref = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(2, 4))
        
        self.count_ua = CountVectorizer(max_features=1000, analyzer='word', ngram_range=(1, 2))
        self.count_uri = CountVectorizer(max_features=800, analyzer='word', ngram_range=(1, 2))
        
        self.svd_ua = TruncatedSVD(n_components=100, random_state=42)
        self.svd_uri = TruncatedSVD(n_components=80, random_state=42)
        
        self.is_fitted = False
        
    def fit_transform(self, texts, text_type='ua'):
        cleaned_texts =[self.clean_text(text) for text in texts]
        if text_type == 'ua':
            tfidf_features = self.tfidf_ua.fit_transform(cleaned_texts)
            tfidf_svd = self.svd_ua.fit_transform(tfidf_features)
            count_features = self.count_ua.fit_transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_svd, count_features, stats_features])
        elif text_type == 'uri':
            tfidf_features = self.tfidf_uri.fit_transform(cleaned_texts)
            tfidf_svd = self.svd_uri.fit_transform(tfidf_features)
            count_features = self.count_uri.fit_transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_svd, count_features, stats_features])
        else:
            tfidf_features = self.tfidf_ref.fit_transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_features, stats_features])
    
    def transform(self, texts, text_type='ua'):
        cleaned_texts = [self.clean_text(text) for text in texts]
        if text_type == 'ua':
            tfidf_features = self.tfidf_ua.transform(cleaned_texts)
            tfidf_svd = self.svd_ua.transform(tfidf_features)
            count_features = self.count_ua.transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_svd, count_features, stats_features])
        elif text_type == 'uri':
            tfidf_features = self.tfidf_uri.transform(cleaned_texts)
            tfidf_svd = self.svd_uri.transform(tfidf_features)
            count_features = self.count_uri.transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_svd, count_features, stats_features])
        else:
            tfidf_features = self.tfidf_ref.transform(cleaned_texts).toarray()
            stats_features = np.array([self.extract_text_stats(text) for text in cleaned_texts])
            return np.hstack([tfidf_features, stats_features])
            
    def extract_text_stats(self, text):
        text = str(text)
        return[
            len(text),
            sum(c.isdigit() for c in text) / max(len(text), 1),
            sum(not c.isalnum() for c in text) / max(len(text), 1),
            len(text.split()),
            len(set(text)),
            max(len(word) for word in text.split()) if text.split() else 0,
        ]
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\/\-\?&=%]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:300]

class EnhancedTextProcessor:
    def __init__(self, max_features=3000):
        self.text_processor = HybridTextProcessor(max_features)
        self.scaler = StandardScaler()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_mask = None
        self.is_fitted = False
        self.ip_stats_map = None  
        
    def prepare_train_features(self, df):

        handcrafted = extract_handcrafted_features(df)
        print("提取时序特征")
        temporal = self.feature_engineer.extract_temporal_features(df)
        print("计算IP统计")
        self.ip_stats_map = self.feature_engineer.compute_ip_stats_for_training(df)
        behavioral = self.feature_engineer.extract_behavioral_features(df, self.ip_stats_map)
        
        print("提取并拟合文本特征")
        ua_texts = df['UserAgent'].fillna('unknown').astype(str)
        uri_texts = df['URI'].fillna('unknown').astype(str)
        ref_texts = df['Referrer'].fillna('unknown').astype(str)
        
        ua_feat = self.text_processor.fit_transform(ua_texts, 'ua')
        uri_feat = self.text_processor.fit_transform(uri_texts, 'uri')
        ref_feat = self.text_processor.fit_transform(ref_texts, 'ref')
        
        all_features = np.hstack([handcrafted, temporal, behavioral, ua_feat, uri_feat, ref_feat])
        print("应用方差筛选和标准化")
        all_features, self.feature_mask = self._variance_threshold_filter(all_features, threshold=0.01)
        all_features = self.scaler.fit_transform(all_features)
        self.is_fitted = True
        labels = np.array([0 if 'human' in str(l).lower() else 1 for l in df['Label']])
        return all_features, labels
    
    def transform_features(self, df):
        if not self.is_fitted:
            raise ValueError("必须先调用prepare_train_features在训练集上fit！")
        handcrafted = extract_handcrafted_features(df)
        temporal = self.feature_engineer.extract_temporal_features(df)
        behavioral = self.feature_engineer.extract_behavioral_features(df, self.ip_stats_map)
        
        ua_texts = df['UserAgent'].fillna('unknown').astype(str)
        uri_texts = df['URI'].fillna('unknown').astype(str)
        ref_texts = df['Referrer'].fillna('unknown').astype(str)
        
        ua_feat = self.text_processor.transform(ua_texts, 'ua')
        uri_feat = self.text_processor.transform(uri_texts, 'uri')
        ref_feat = self.text_processor.transform(ref_texts, 'ref')
        
        all_features = np.hstack([handcrafted, temporal, behavioral, ua_feat, uri_feat, ref_feat])
        if self.feature_mask is not None:
            all_features = all_features[:, self.feature_mask]
        all_features = self.scaler.transform(all_features)
        labels = np.array([0 if 'human' in str(l).lower() else 1 for l in df['Label']])
        return all_features, labels

    def _variance_threshold_filter(self, features, threshold=0.01):
        variances = np.var(features, axis=0)
        mask = variances > threshold
        return features[:, mask], mask


class GatedResidualNetwork(nn.Module):
    """门控残差网络"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 门控机制
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        hidden = self.elu(self.linear1(x))
        hidden = self.dropout(self.linear2(hidden))
        # 门控输出激活，控制信息流
        gate = torch.sigmoid(self.gate(hidden))
        # 引入残差连接
        return self.norm(skip + gate * hidden)


class SparseMoEBlock(nn.Module):
    """稀疏混合专家网络"""
    def __init__(self, dim, num_experts=4, top_k=2, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        # 路由网络
        self.router = nn.Linear(dim, num_experts)
        # 专家库
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 计算路由概率
        routing_logits = self.router(x)
        routing_probs = F.softmax(routing_logits, dim=-1) # [B, num_experts]
        
        # 选出 Top-K 专家
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1) 
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True) # 重新归一化
        
        # 为了高效实现且避免复杂的动态索引，并行计算所有专家结果后使用掩码进行软路由
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [B, num_experts, dim]
        
        # 构建仅保留 Top-K 的掩码
        mask = torch.zeros_like(routing_probs).scatter_(1, top_k_indices, 1.0)
        masked_probs = routing_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 加权融合专家输出
        output = torch.sum(expert_outputs * masked_probs.unsqueeze(-1), dim=1) #[B, dim]
        return self.norm(x + output)


class OrthogonalPrototypeDisentanglement(nn.Module):
    """正交原型特征解耦"""
    def __init__(self, input_dim, latent_dim=128, num_prototypes=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.human_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.bot_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # 可学习的代表性行为锚点
        self.human_prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        self.bot_prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        
        # 初始化正交化，确保原型空间的彼此独立性
        nn.init.orthogonal_(self.human_prototypes)
        nn.init.orthogonal_(self.bot_prototypes)

    def forward(self, x):
        z_human = self.human_proj(x) 
        z_bot = self.bot_proj(x)    
        
        # 与原型的相似度计算
        sim_human = torch.matmul(z_human, self.human_prototypes.t()) / (self.latent_dim ** 0.5) 
        attn_human = F.softmax(sim_human, dim=-1)
        # 用原型重建人类特征
        recon_human = torch.matmul(attn_human, self.human_prototypes)
        human_features = z_human + recon_human # 引入残差
        
        sim_bot = torch.matmul(z_bot, self.bot_prototypes.t()) / (self.latent_dim ** 0.5)
        attn_bot = F.softmax(sim_bot, dim=-1)
        recon_bot = torch.matmul(attn_bot, self.bot_prototypes)
        bot_features = z_bot + recon_bot
        
        return human_features, bot_features


class AdvancedBotDetectorNN(nn.Module):
    def __init__(self, input_dim, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'hidden_dim': 256,
                'latent_dim': 128,
                'num_heads': 4,
                'dropout': 0.3
            }
        
        self.feature_mapping = GatedResidualNetwork(
            input_dim, config['hidden_dim'], dropout=config['dropout']
        )
        
        # 行为模式路由层
        self.moe_block = SparseMoEBlock(
            config['hidden_dim'], num_experts=4, top_k=2, dropout=config['dropout']
        )
        
        # 特征子空间自注意力机制
        self.num_subspaces = 8
        self.subspace_dim = config['hidden_dim'] // self.num_subspaces
        self.subspace_attention = nn.MultiheadAttention(
            embed_dim=self.subspace_dim,
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config['hidden_dim'])
        
        # 原型解耦机制
        self.disentangle = OrthogonalPrototypeDisentanglement(
            config['hidden_dim'],
            latent_dim=config['latent_dim'],
            num_prototypes=8
        )
        
        # 独立分类器
        self.human_classifier = nn.Sequential(
            nn.Linear(config['latent_dim'], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1)
        )
        
        self.bot_classifier = nn.Sequential(
            nn.Linear(config['latent_dim'], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1)
        )
        
        # 双线性协方差融合
        self.bilinear_fusion = nn.Bilinear(config['latent_dim'], config['latent_dim'], 64)
        self.fusion_classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x_mapped = self.feature_mapping(x)
        x_moe = self.moe_block(x_mapped)
        
        # 特征子空间自注意力交互
        B = x_moe.size(0)
        x_seq = x_moe.view(B, self.num_subspaces, self.subspace_dim)
        attn_out, _ = self.subspace_attention(x_seq, x_seq, x_seq)
        x_attn = attn_out.contiguous().view(B, -1) 
        x_interact = self.attn_norm(x_moe + x_attn)
        
        # 原型解耦表示
        human_features, bot_features = self.disentangle(x_interact)
        
        # 计算 Logits
        human_logits = self.human_classifier(human_features)
        bot_logits = self.bot_classifier(bot_features)
        
        # 双线性融合机制
        fused_hidden = self.bilinear_fusion(human_features, bot_features)
        fused_logits = self.fusion_classifier(fused_hidden)
        
        # 自适应集成
        final_logits = (human_logits + bot_logits + fused_logits) / 3
        
        # 严格保持与训练循环的接口兼容性
        return {
            'logits': final_logits,
            'human_features': human_features,
            'bot_features': bot_features
        }


class FeatureNoiseTransform:
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
    def __call__(self, x):
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

class BotDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

class ImprovedBotTrainer:
    def __init__(self, device=device):
        self.device = device
        self.model = None
        self.text_processor = None  
        self.best_metrics = {}
        self.input_dim = None
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

    def create_weighted_sampler(self, labels):
        class_counts = np.bincount(labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    def train_model(self, train_df, val_df, epochs=50, batch_size=128, learning_rate=1e-3):
        self.text_processor = EnhancedTextProcessor()
        
        X_train, y_train = self.text_processor.prepare_train_features(train_df)
        X_train = np.nan_to_num(X_train)
        
        X_val, y_val = self.text_processor.transform_features(val_df)
        X_val = np.nan_to_num(X_val)
        
        self.input_dim = X_train.shape[1]
        
        train_dataset = BotDataset(X_train, y_train, transform=FeatureNoiseTransform())
        val_dataset = BotDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                sampler=self.create_weighted_sampler(y_train), 
                                num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)

        self.model = AdvancedBotDetectorNN(
            input_dim=self.input_dim,
            config={'hidden_dim': 256, 'latent_dim': 128, 'num_heads': 4, 'dropout': 0.3}
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        best_val_auc = 0
        best_model_state = None

        print(f"开始训练 (Epochs: {epochs})...")
        for epoch in range(epochs):
            self.model.train()
            train_loss_accum = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                cls_loss = self.cls_loss_fn(outputs['logits'].squeeze(), batch_labels)
                contrastive_target = torch.ones(batch_features.size(0), device=self.device) * -1
                cont_loss = self.contrastive_loss_fn(
                    outputs['human_features'], 
                    outputs['bot_features'],
                    contrastive_target
                )
                
                # BCE 分类损失与正交解耦损失联合优化
                loss = cls_loss + 0.1 * cont_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss_accum += loss.item()

            val_preds, val_true = self._predict_loader(self.model, val_loader)
            try:
                val_auc = roc_auc_score(val_true, val_preds)
            except:
                val_auc = 0.5 
            
            scheduler.step(val_auc)

            print(f'Epoch {epoch+1}/{epochs}: Loss {train_loss_accum/len(train_loader):.4f}, Val AUC {val_auc:.4f}')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        print(f"训练完成 最佳验证 AUC: {best_val_auc:.4f}")
        return self.model

    def _predict_loader(self, model, data_loader):
        model.eval()
        probs_list = []
        true_list =[]
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                outputs = model(features)
                probs = torch.sigmoid(outputs['logits']).squeeze()
                probs_list.extend(probs.cpu().numpy())
                true_list.extend(labels.numpy())
        return np.array(probs_list), np.array(true_list)

    def evaluate(self, test_df):
        if self.model is None: 
            raise ValueError("请先训练模型")
        if self.text_processor is None or not self.text_processor.is_fitted:
            raise ValueError("特征处理器未初始化")
        
        features, labels = self.text_processor.transform_features(test_df)
        features = np.nan_to_num(features)
        
        dataset = BotDataset(features, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        probabilities, true_labels = self._predict_loader(self.model, loader)
        predictions = (probabilities > 0.5).astype(int)
        
        print("\n分类报告:")
        print(classification_report(true_labels, predictions, target_names=['Human', 'Bot']))
        
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        auc_roc = roc_auc_score(true_labels, probabilities)
        
        print(f"F1分数: {f1:.4f} | AUC-ROC: {auc_roc:.4f}")
        
        self.best_metrics.update({'f1': f1, 'auc_roc': auc_roc, 'recall': recall, 'precision': precision})
        return predictions, probabilities, true_labels


def clean_data(df):
    def safe_convert_bytes(x):
        try: 
            return int(float(str(x).strip())) if str(x).strip() not in ['', '-'] else 0
        except: 
            return 0
    
    df['Bytes'] = df['Bytes'].apply(safe_convert_bytes)
    df.fillna({'UserAgent': 'unknown', 'Referrer': 'unknown', 'URI': 'unknown', 'IP': '0.0.0.0'}, inplace=True)
    df['StatusCode'] = df['StatusCode'].fillna(200)
    
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    if df['TimeStamp'].isna().any():
        df.loc[df['TimeStamp'].isna(), 'TimeStamp'] = pd.Timestamp.now()
        
    df['Label'] = df['Label'].fillna('Human').astype(str)
    return df

def main():
    try:
        start_time = time.time()
        train_val_df = pd.read_csv('../train.csv')
        test_df = pd.read_csv('../test.csv')
        
        expected_cols =['IP', 'TimeStamp', 'RequestMethod', 'URI', 'Protocol', 'StatusCode', 'Bytes', 'Referrer', 'UserAgent', 'Label']
        if len(train_val_df.columns) >= 10: 
            train_val_df.columns = expected_cols[:len(train_val_df.columns)]
        if len(test_df.columns) >= 10:
            test_df.columns = expected_cols[:len(test_df.columns)]
        
        train_val_df = clean_data(train_val_df)
        test_df = clean_data(test_df)
        
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['Label'])
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
              
        trainer = ImprovedBotTrainer(device)
        trainer.train_model(train_df, val_df, epochs=100, batch_size=512)
        
        trainer.evaluate(test_df)
        
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'feature_dim': trainer.input_dim,
            'model_config': {'hidden_dim': 256, 'latent_dim': 128, 'num_heads': 4, 'dropout': 0.3}
        }, 'advanced_bot_detector_fixed.pth')
        
        joblib.dump(trainer.text_processor, 'text_processor_fixed.pkl')
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()