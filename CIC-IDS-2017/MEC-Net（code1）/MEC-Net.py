import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.feature_selection import VarianceThreshold

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'

set_seeds(42)

# ===================== 多专家认知门控特征重组模块 =====================
class CognitiveEnhancedFeatureReorganization(tf.keras.layers.Layer):
    """认知引导的动态特征重组,模拟人类认知过程中的信息整合 (CEFR)"""
    def __init__(self, latent_dim=128, num_experts=4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_experts = num_experts

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]

        self.experts = [self.add_weight(
            shape=(self.feature_dim, self.latent_dim),
            initializer=tf.keras.initializers.Orthogonal(seed=42),
            trainable=True,
            name=f'expert_{i}'
        ) for i in range(self.num_experts)]

        self.gating_net = tf.keras.Sequential([
            Dense(256, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            Dense(self.num_experts, activation='softmax',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.fusion_net = tf.keras.Sequential([
            Dense(256, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(self.feature_dim, activation=None,
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.adaptive_weight = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.7),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9)
        )
        super().build(input_shape)

    def call(self, inputs):
        expert_outputs = []
        for expert in self.experts:
            transformed = tf.einsum('btd,dl->btl', inputs, expert)
            expert_outputs.append(transformed)

        expert_stack = tf.stack(expert_outputs, axis=1)
        context = tf.reduce_mean(inputs, axis=1)
        gating_weights = self.gating_net(context)
        gating_weights = tf.expand_dims(gating_weights, axis=-1)
        gating_weights = tf.expand_dims(gating_weights, axis=-1)

        fused_representation = tf.reduce_sum(expert_stack * gating_weights, axis=1)
        fusion_output = self.fusion_net(fused_representation)
        result = (self.adaptive_weight * inputs + (1 - self.adaptive_weight) * fusion_output)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


# ===================== 层级自适应注意力融合机制 =====================
class SelfEvolvingHierarchicalAttention(tf.keras.layers.Layer):
    """自适应多粒度注意力机制 (SEHA)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]

        self.feat_att = tf.keras.Sequential([
            Dense(128, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(self.feat_dim, activation='sigmoid',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.group_num = min(8, self.feat_dim)

        self.group_att = tf.keras.Sequential([
            Dense(128, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(1, activation='sigmoid',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.fusion_gate = tf.keras.Sequential([
            Dense(128, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(3, activation='softmax',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])
        super().build(input_shape)

    def call(self, inputs):
        feat_att = self.feat_att(tf.reduce_mean(inputs, axis=1))
        feat_out = inputs * tf.expand_dims(feat_att, axis=1)

        if self.feat_dim < self.group_num:
            return feat_out

        group_size = self.feat_dim // self.group_num
        remainder = self.feat_dim % self.group_num
        group_sizes = [group_size] * self.group_num
        if remainder > 0:
            group_sizes[-1] += remainder

        groups = tf.split(inputs, group_sizes, axis=-1)
        group_outs = []
        for group in groups:
            group_mean = tf.reduce_mean(group, axis=-1, keepdims=True)
            group_att = self.group_att(group_mean)
            group_out = group * group_att
            group_outs.append(group_out)

        grouped_out = tf.concat(group_outs, axis=-1)
        channel_weights = tf.reduce_mean(inputs, axis=1, keepdims=True)
        channel_out = inputs * channel_weights

        gate_vals = self.fusion_gate(tf.reduce_mean(inputs, axis=1))
        g1, g2, g3 = tf.unstack(gate_vals, axis=-1)

        result = (g1[:, None, None] * feat_out +
                  g2[:, None, None] * grouped_out +
                  g3[:, None, None] * channel_out)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


# ===================== 元认知风险感知残差学习 =====================
class MetaCognitiveGuidedResidual(tf.keras.layers.Layer):
    """结合元认知概念的残差单元,模拟人类学习的反思过程 (MCGR)"""
    def __init__(self, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]

        self.meta_cognitive_evaluator = tf.keras.Sequential([
            Dense(128, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(1, activation='sigmoid',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.residual_path = tf.keras.Sequential([
            Dense(self.feat_dim * 2, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dropout(self.dropout_rate, seed=42),
            Dense(self.feat_dim, activation=None,
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.adaptive_weight = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9)
        )
        super().build(input_shape)

    def call(self, inputs):
        meta_output = self.meta_cognitive_evaluator(tf.reduce_mean(inputs, axis=1))
        risk_index = meta_output
        adjustment_factor = tf.sqrt(tf.maximum(risk_index, 1e-8))
        adjustment_factor = tf.clip_by_value(adjustment_factor, 0.1, 2.0)
        adjustment_factor = tf.expand_dims(adjustment_factor, axis=1)

        residual = self.residual_path(inputs)
        residual = tf.clip_by_value(residual, -10.0, 10.0)
        output = inputs + adjustment_factor * residual * self.adaptive_weight
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


# ===================== 双向特征融合桥梁 =====================
class BidirectionalFeatureFusionBridge(tf.keras.layers.Layer):
    """连接不同网络层级的语义蒸馏桥梁,实现知识迁移 (BFFB)"""
    def __init__(self, distillation_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.distillation_dim = distillation_dim

    def build(self, input_shapes):
        self.early_feat_dim = input_shapes[0][-1]
        self.late_feat_dim = input_shapes[1][-1]

        self.early_processor = tf.keras.Sequential([
            Dense(self.distillation_dim, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6)
        ])

        self.late_processor = tf.keras.Sequential([
            Dense(self.distillation_dim, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6)
        ])

        self.distiller = tf.keras.Sequential([
            Dense(self.distillation_dim * 2, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dropout(0.3, seed=42),
            Dense(self.distillation_dim, activation='linear',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.fusion_gate = Dense(1, activation='sigmoid',
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                                 kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        super().build(input_shapes)

    def call(self, inputs):
        early_features, late_features = inputs
        processed_early = self.early_processor(early_features)
        processed_late = self.late_processor(late_features)
        distilled = self.distiller(tf.concat([processed_early, processed_late], axis=-1))
        fusion_weight = self.fusion_gate(tf.reduce_mean(distilled, axis=1))
        final_features = (fusion_weight[:, None, :] * distilled +
                          (1 - fusion_weight[:, None, :]) * processed_late)
        return final_features

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], self.distillation_dim)


# ===================== 自适应多尺度特征融合 =====================
class AdaptiveScaleFeaturePyramidFusion(tf.keras.layers.Layer):
    """自适应多尺度特征融合 (ASFPF)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]
        self.compressed_dim = self.feat_dim

        self.global_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6)
        ])

        self.local_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6)
        ])

        self.context_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6)
        ])

        self.fusion_gate = tf.keras.Sequential([
            Dense(128, activation='swish',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0)),
            LayerNormalization(epsilon=1e-6),
            Dense(3, activation='softmax',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                  kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
        ])

        self.feature_reconstructor = Dense(
            self.feat_dim, activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            kernel_constraint=tf.keras.constraints.MaxNorm(4.0)
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]

        global_feat = tf.reduce_mean(inputs, axis=1, keepdims=True)
        global_feat = tf.tile(global_feat, [1, seq_len, 1])
        global_processed = self.global_path(global_feat)

        local_processed = self.local_path(inputs)

        half_point = tf.maximum(tf.cast(seq_len // 2, tf.int32), 1)
        quarter_point = tf.maximum(tf.cast(seq_len // 4, tf.int32), 1)

        half_mean = tf.reduce_mean(inputs[:, :half_point], axis=1, keepdims=True)
        quarter_mean = tf.reduce_mean(inputs[:, :quarter_point], axis=1, keepdims=True)

        context_feat = tf.concat([half_mean, quarter_mean], axis=-1)
        context_feat = tf.tile(context_feat, [1, seq_len, 1])
        context_processed = self.context_path(context_feat)

        gate_input = tf.reduce_mean(inputs, axis=1)
        gate_weights = self.fusion_gate(gate_input)
        g1, g2, g3 = tf.unstack(gate_weights, axis=-1)

        fused = (g1[:, None, None] * global_processed +
                 g2[:, None, None] * local_processed +
                 g3[:, None, None] * context_processed)

        reconstructed = self.feature_reconstructor(fused)
        reconstructed = tf.clip_by_value(reconstructed, -10.0, 10.0)
        result = inputs + reconstructed
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


# ===================== 进化认知网络架构 =====================
def build_ec_net(input_shape, num_classes,
                 use_cefr=True, use_seha=True,
                 use_asfpf=True, use_mcgr=True,
                 use_bffb=True,
                 input_noise_std=0.02,
                 inter_module_dropout=0.15,
                 head_dropout1=0.3,
                 head_dropout2=0.2,
                 post_concat_dropout=0.2):
                     
    inputs = tf.keras.Input(shape=input_shape)

    if input_noise_std > 0:
        x = GaussianNoise(input_noise_std)(inputs)
    else:
        x = inputs

    if use_cefr:
        x1 = CognitiveEnhancedFeatureReorganization(num_experts=2)(x)
    else:
        x1 = x
    x1 = LayerNormalization(epsilon=1e-6)(x1)
    if inter_module_dropout > 0:
        x1 = Dropout(inter_module_dropout, seed=42)(x1)

    if use_seha:
        x2 = SelfEvolvingHierarchicalAttention()(x1)
    else:
        x2 = x1
    x2 = LayerNormalization(epsilon=1e-6)(x2)
    if inter_module_dropout > 0:
        x2 = Dropout(inter_module_dropout, seed=42)(x2)

    if use_asfpf:
        x3 = AdaptiveScaleFeaturePyramidFusion()(x2)
    else:
        x3 = x2
    x3 = LayerNormalization(epsilon=1e-6)(x3)
    if inter_module_dropout > 0:
        x3 = Dropout(inter_module_dropout, seed=42)(x3)

    if use_mcgr:
        x4 = MetaCognitiveGuidedResidual()(x3)
    else:
        x4 = x3
    x4 = LayerNormalization(epsilon=1e-6)(x4)
    if inter_module_dropout > 0:
        x4 = Dropout(inter_module_dropout, seed=42)(x4)

    early_features = x4

    x = x4
    for i in range(2):
        if use_mcgr:
            x = MetaCognitiveGuidedResidual()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Activation('swish')(x)
        if inter_module_dropout > 0:
            x = Dropout(inter_module_dropout, seed=42)(x)

    if use_bffb:
        distilled = BidirectionalFeatureFusionBridge()([early_features, x])
        distilled = LayerNormalization(epsilon=1e-6)(distilled)
    else:
        distilled = Dense(128, activation='swish',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(x)
        distilled = LayerNormalization(epsilon=1e-6)(distilled)
    if inter_module_dropout > 0:
        distilled = Dropout(inter_module_dropout, seed=42)(distilled)

    gap = GlobalAveragePooling1D()(distilled)
    gmp = GlobalMaxPooling1D()(distilled)
    std = tf.keras.layers.Lambda(
        lambda t: tf.sqrt(tf.math.reduce_variance(t, axis=1) + 1e-8)
    )(distilled)
    concat = Concatenate()([gap, gmp, std])

    if post_concat_dropout > 0:
        concat = Dropout(post_concat_dropout, seed=42)(concat)

    x = Dense(512, activation='swish',
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
              kernel_constraint=tf.keras.constraints.MaxNorm(4.0))(concat)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(head_dropout1, seed=42)(x)

    x = Dense(256, activation='swish',
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
              kernel_constraint=tf.keras.constraints.MaxNorm(4.0))(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(head_dropout2, seed=42)(x)

    outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ===================== 标签平滑损失 =====================
def make_smoothed_sparse_crossentropy(num_classes, label_smoothing=0.01):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        y_true_smooth = (y_true_oh * (1.0 - label_smoothing)
                         + label_smoothing / tf.cast(num_classes, tf.float32))
        return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
    return loss_fn


# ===================== 训练监控器 =====================
class CognitiveTrainingMonitor(tf.keras.callbacks.Callback):
    """训练过程监控器"""
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.best_weights = None
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy', 0)
        if current_val_acc > self.best_val_accuracy:
            self.best_val_accuracy = current_val_acc
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            print(f"新的最佳验证准确率: {self.best_val_accuracy:.4f}")

        if np.isnan(logs.get('loss', 0)) or np.isinf(logs.get('loss', 0)):
            print("检测到NaN或Inf,停止训练")
            self.model.stop_training = True


def load_enhanced_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    max_valid_value = 1e20
    min_valid_value = -1e20

    X_train = np.clip(X_train, min_valid_value, max_valid_value)
    X_test = np.clip(X_test, min_valid_value, max_valid_value)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test, scaler


def train_model(X_train_main, y_train_main, X_val, y_val, X_test, y_test,
                num_classes, feat_dim,
                use_cefr=True, use_seha=True, use_asfpf=True,
                use_mcgr=True, use_bffb=True,
                epochs=150, batch_size=256, verbose=2,
                label_smoothing=0.01):
    set_seeds(42)

    model = build_ec_net(
        input_shape=X_train_main.shape[1:],
        num_classes=num_classes,
        use_cefr=use_cefr, use_seha=use_seha,
        use_asfpf=use_asfpf, use_mcgr=use_mcgr,
        use_bffb=use_bffb
    )

    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=1e-4,
        clipnorm=1.0
    )

    loss_fn = make_smoothed_sparse_crossentropy(num_classes, label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3),
            tf.keras.metrics.SparseCategoricalCrossentropy(name='xe_loss')
        ]
    )

    cognitive_monitor = CognitiveTrainingMonitor(validation_data=(X_val, y_val))

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_main, y_train_main))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_main), seed=42)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[cognitive_monitor, reduce_lr, early_stopping],
        verbose=verbose
    )

    y_pred_raw = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_raw, axis=1)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred_classes, average='weighted', zero_division=0
    )

    return precision, recall, fscore


def main():
    set_seeds(42)

    try:
        X_train, y_train, X_test, y_test, label_encoder = load_enhanced_data(
            '../train.csv', '../test.csv'
        )
        num_classes = len(np.unique(np.concatenate([y_train, y_test])))

        SEQ_LEN = 2
        n_features = X_train.shape[-1]
        chunk_dim = int(np.ceil(n_features / SEQ_LEN))
        padded_dim = chunk_dim * SEQ_LEN
        pad_size = padded_dim - n_features

        if pad_size > 0:
            X_train_pad = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
            X_test_pad = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        else:
            X_train_pad, X_test_pad = X_train, X_test

        X_train_3d = X_train_pad.reshape(-1, SEQ_LEN, chunk_dim).astype(np.float32)
        X_test_3d = X_test_pad.reshape(-1, SEQ_LEN, chunk_dim).astype(np.float32)


        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train_3d, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        feat_dim = X_train_3d.shape[-1]

        model_full = build_ec_net(
            input_shape=X_train_3d.shape[1:],
            num_classes=num_classes,
            use_cefr=True, use_seha=True,
            use_asfpf=True, use_mcgr=True, use_bffb=True
        )
        model_full.summary()

        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4, clipnorm=1.0)

        LABEL_SMOOTHING = 0.001
        loss_fn = make_smoothed_sparse_crossentropy(num_classes, label_smoothing=LABEL_SMOOTHING)

        model_full.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='xe_loss')
            ]
        )

        cognitive_monitor = CognitiveTrainingMonitor(validation_data=(X_val, y_val))

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=100, restore_best_weights=True, verbose=1
        )

        train_dataset_full = tf.data.Dataset.from_tensor_slices((X_train_main, y_train_main))
        train_dataset_full = train_dataset_full.shuffle(buffer_size=len(X_train_main), seed=42)

        train_dataset_full = train_dataset_full.batch(128)
        train_dataset_full = train_dataset_full.prefetch(tf.data.AUTOTUNE)

        history = model_full.fit(
            train_dataset_full,
            epochs=300,
            validation_data=(X_val, y_val),
            callbacks=[cognitive_monitor, reduce_lr, early_stopping],
            verbose=2
        )

        test_results = model_full.evaluate(X_test_3d, y_test, verbose=0)

        y_pred_raw = model_full.predict(X_test_3d, verbose=0)
        y_pred_classes = np.argmax(y_pred_raw, axis=1)

        p_full, r_full, f_full, _ = precision_recall_fscore_support(
            y_test, y_pred_classes, average='weighted', zero_division=0
        )

        print("\n=== 完整模型评估结果 ===")
        print(f"加权精确率: {p_full:.4f}")
        print(f"加权召回率: {r_full:.4f}")
        print(f"加权F1分数: {f_full:.4f}")

        print("\n=== 详细分类报告 ===")
        print(classification_report(y_test, y_pred_classes,
                                    target_names=[f'Class_{i}' for i in range(num_classes)]))
        model_full.save_weights('ec_net_model_weights5.h5')

        return model_full, history, test_results

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    trained_model, training_history, final_results = main()