import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout, Activation, GaussianNoise,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.optimizers import AdamW
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(seed)
    os.environ.update({
        'TF_DETERMINISTIC_OPS': '1',
        'TF_CUDNN_DETERMINISTIC': '1',
        'PYTHONHASHSEED': str(seed),
        'OMP_NUM_THREADS': '1',
    })

set_seeds(42)


# ===================== 多专家认知门控特征重组模块 (CEFR) =====================
class CognitiveEnhancedFeatureReorganization(tf.keras.layers.Layer):
    """认知引导的动态特征重组，模拟人类认知过程中的信息整合 (CEFR)"""
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
        result = (self.adaptive_weight * inputs +
                  (1 - self.adaptive_weight) * fusion_output)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


# ===================== 层级自适应注意力融合机制 (SEHA) =====================
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


# ===================== 元认知风险感知残差学习 (MCGR) =====================
class MetaCognitiveGuidedResidual(tf.keras.layers.Layer):
    """结合元认知概念的残差单元，模拟人类学习的反思过程 (MCGR)"""
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


# ===================== 双向特征融合桥梁 (BFFB) =====================
class BidirectionalFeatureFusionBridge(tf.keras.layers.Layer):
    """连接不同网络层级的语义蒸馏桥梁，实现知识迁移 (BFFB)"""
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


# ===================== 自适应多尺度特征融合 (ASFPF) =====================
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


def build_teacher_extractor(teacher: tf.keras.Model):
    bridge_layers = [l for l in teacher.layers
                     if isinstance(l, BidirectionalFeatureFusionBridge)]
    dense256_layers = [l for l in teacher.layers
                       if isinstance(l, Dense) and l.units == 256]
    outs = {}
    if bridge_layers:
        outs['bridge'] = bridge_layers[-1].output
    if dense256_layers:
        outs['pre_logit'] = dense256_layers[-1].output
    extractor = tf.keras.Model(inputs=teacher.input, outputs=outs,
                               name='TeacherExtractor')
    extractor.trainable = False
    return extractor


# ============================ 学生轻量化模块 ================================
class LightweightSEHA(tf.keras.layers.Layer):
    """轻量化 SEHA"""
    def __init__(self, inner_dim=32, group_num=4, **kwargs):
        super().__init__(**kwargs)
        self.inner_dim = inner_dim
        self.group_num = group_num

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]
        self.actual_group_num = min(self.group_num, self.feat_dim)
        kw = dict(kernel_constraint=tf.keras.constraints.MaxNorm(4.0))

        self.feat_att = tf.keras.Sequential([
            Dense(self.inner_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dense(self.feat_dim, activation='sigmoid', **kw)
        ])
        self.group_att = tf.keras.Sequential([
            Dense(self.inner_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dense(1, activation='sigmoid', **kw)
        ])
        self.fusion_gate = tf.keras.Sequential([
            Dense(self.inner_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dense(3, activation='softmax', **kw)
        ])
        super().build(input_shape)

    def call(self, inputs):
        feat_att = self.feat_att(tf.reduce_mean(inputs, axis=1))
        feat_out = inputs * tf.expand_dims(feat_att, axis=1)

        if self.feat_dim < self.actual_group_num:
            return feat_out

        group_size = self.feat_dim // self.actual_group_num
        remainder = self.feat_dim % self.actual_group_num
        group_sizes = [group_size] * self.actual_group_num
        if remainder > 0:
            group_sizes[-1] += remainder

        groups = tf.split(inputs, group_sizes, axis=-1)
        group_outs = []
        for group in groups:
            gm = tf.reduce_mean(group, axis=-1, keepdims=True)
            ga = self.group_att(gm)
            group_outs.append(group * ga)
        grouped_out = tf.concat(group_outs, axis=-1)

        channel_out = inputs * tf.reduce_mean(inputs, axis=1, keepdims=True)

        gate_vals = self.fusion_gate(tf.reduce_mean(inputs, axis=1))
        g1, g2, g3 = tf.unstack(gate_vals, axis=-1)
        return (g1[:, None, None] * feat_out +
                g2[:, None, None] * grouped_out +
                g3[:, None, None] * channel_out)

    def compute_output_shape(self, input_shape):
        return input_shape


class LightweightMCGR(tf.keras.layers.Layer):
    """轻量化 MCGR"""
    def __init__(self, dropout_rate=0.1, inner_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]
        kw = dict(kernel_constraint=tf.keras.constraints.MaxNorm(4.0))

        self.meta_cognitive_evaluator = tf.keras.Sequential([
            Dense(self.inner_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dense(1, activation='sigmoid', **kw)
        ])
        self.residual_path = tf.keras.Sequential([
            Dense(self.feat_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dropout(self.dropout_rate, seed=42),
            Dense(self.feat_dim, activation=None, **kw)
        ])
        self.adaptive_weight = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9))
        super().build(input_shape)

    def call(self, inputs):
        risk = self.meta_cognitive_evaluator(tf.reduce_mean(inputs, axis=1))
        adj = tf.clip_by_value(tf.sqrt(tf.maximum(risk, 1e-8)), 0.1, 2.0)
        adj = tf.expand_dims(adj, axis=1)
        res = tf.clip_by_value(self.residual_path(inputs), -10.0, 10.0)
        return inputs + adj * res * self.adaptive_weight

    def compute_output_shape(self, input_shape):
        return input_shape


class LightweightASFPF(tf.keras.layers.Layer):
    """轻量化 ASFPF"""
    def __init__(self, compress_ratio=4, inner_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.compress_ratio = compress_ratio
        self.inner_dim = inner_dim

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]
        self.compressed_dim = max(16, self.feat_dim // self.compress_ratio)
        kw = dict(kernel_constraint=tf.keras.constraints.MaxNorm(4.0))

        self.global_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6)
        ])
        self.local_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6)
        ])
        self.context_path = tf.keras.Sequential([
            Dense(self.compressed_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6)
        ])
        self.fusion_gate = tf.keras.Sequential([
            Dense(self.inner_dim, activation='swish', **kw),
            LayerNormalization(epsilon=1e-6),
            Dense(3, activation='softmax', **kw)
        ])
        self.feature_reconstructor = Dense(self.feat_dim, activation=None, **kw)
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        global_feat = tf.tile(tf.reduce_mean(inputs, axis=1, keepdims=True), [1, seq_len, 1])
        global_processed = self.global_path(global_feat)
        local_processed = self.local_path(inputs)
        half_point = tf.maximum(tf.cast(seq_len // 2, tf.int32), 1)
        quarter_point = tf.maximum(tf.cast(seq_len // 4, tf.int32), 1)
        half_mean = tf.reduce_mean(inputs[:, :half_point], axis=1, keepdims=True)
        quarter_mean = tf.reduce_mean(inputs[:, :quarter_point], axis=1, keepdims=True)
        context_feat = tf.tile(tf.concat([half_mean, quarter_mean], axis=-1), [1, seq_len, 1])
        context_processed = self.context_path(context_feat)
        gate_weights = self.fusion_gate(tf.reduce_mean(inputs, axis=1))
        g1, g2, g3 = tf.unstack(gate_weights, axis=-1)
        fused = (g1[:, None, None] * global_processed +
                 g2[:, None, None] * local_processed +
                 g3[:, None, None] * context_processed)
        reconstructed = tf.clip_by_value(self.feature_reconstructor(fused), -10.0, 10.0)
        return inputs + reconstructed

    def compute_output_shape(self, input_shape):
        return input_shape


def build_student(input_shape, num_classes, bridge_dim: int = 32):
    """
    极致轻量化学生模型
    """
    kw = dict(kernel_constraint=tf.keras.constraints.MaxNorm(4.0))
    inp = tf.keras.Input(shape=input_shape, name='student_input')

    x1 = CognitiveEnhancedFeatureReorganization(latent_dim=16, num_experts=2)(inp)
    x1 = LayerNormalization(epsilon=1e-6)(x1)

    x2 = LightweightSEHA(inner_dim=32, group_num=4)(x1)
    x2 = LayerNormalization(epsilon=1e-6)(x2)

    x3 = LightweightASFPF(compress_ratio=4, inner_dim=32)(x2)
    x3 = LayerNormalization(epsilon=1e-6)(x3)

    x4 = LightweightMCGR(dropout_rate=0.10, inner_dim=32)(x3)
    x4 = LayerNormalization(epsilon=1e-6)(x4)
    x4 = LightweightMCGR(dropout_rate=0.05, inner_dim=32)(x4)
    x4 = LayerNormalization(epsilon=1e-6)(x4)

    dist_s = BidirectionalFeatureFusionBridge(distillation_dim=bridge_dim)([x4, x4])
    dist_s = LayerNormalization(epsilon=1e-6)(dist_s)

    gap = GlobalAveragePooling1D()(dist_s)            
    gmp = GlobalMaxPooling1D()(dist_s)  
    std_s = tf.keras.layers.Lambda(
        lambda t: tf.sqrt(tf.math.reduce_variance(t, axis=1) + 1e-8)
    )(dist_s)                        
    cat = Concatenate()([gap, gmp, std_s])    

    bridge_gap = gap  

    x = Dense(64, activation='swish', **kw)(cat)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.15, seed=42)(x)
    x = Dense(32, activation='swish', **kw)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    logits  = Dense(num_classes, name='s_logits')(x)
    outputs = tf.keras.layers.Softmax(name='s_output')(logits)

    model_train = tf.keras.Model(
        inputs=inp,
        outputs={
            'probs':      outputs,
            'logits':     logits,
            'mid':        cat,
            'bridge_gap': bridge_gap,
        },
        name='StudentTrain_v4')
    model_infer = tf.keras.Model(inputs=inp, outputs=outputs, name='Student_v4')
    return model_train, model_infer


# ============================ 蒸馏组件 ============================
class SDAT(tf.keras.layers.Layer):
    """逐样本自适应蒸馏温度：基于教师预测熵动态调整温度"""
    def __init__(self, T_min=2.0, T_max=8.0, **kwargs):
        super().__init__(**kwargs)
        self.T_min = T_min
        self.T_max = T_max

    def call(self, teacher_logits):
        probs = tf.nn.softmax(teacher_logits, axis=-1)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
        C = tf.cast(tf.shape(teacher_logits)[-1], tf.float32)
        norm_H = tf.clip_by_value(entropy / (tf.math.log(C) + 1e-8), 0.0, 1.0)
        return self.T_min + (self.T_max - self.T_min) * norm_H


class ICD(tf.keras.layers.Layer):
    """反向相关性蒸馏：对齐师生样本间关系结构矩阵"""
    def __init__(self, tau=0.05, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def call(self, feat_t, feat_s):
        ft = tf.math.l2_normalize(feat_t, axis=-1)
        fs = tf.math.l2_normalize(feat_s, axis=-1)
        R_T = tf.matmul(ft, ft, transpose_b=True) / self.tau
        R_S = tf.matmul(fs, fs, transpose_b=True) / self.tau
        mask = 1.0 - tf.eye(tf.shape(ft)[0])
        R_T = R_T * mask
        R_S = R_S * mask
        p_T = tf.nn.softmax(R_T, axis=-1)
        log_p_S = tf.nn.log_softmax(R_S, axis=-1)
        return tf.reduce_mean(
            tf.reduce_sum(p_T * (tf.math.log(p_T + 1e-10) - log_p_S), axis=-1))


class LSTR(tf.keras.layers.Layer):
    """层级软标签再生：从教师 bridge 特征再生软目标分布"""
    def __init__(self, num_classes: int, hidden_dim: int = 64,
                 soft_temp: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self.soft_temp   = soft_temp
        self.dense1 = Dense(hidden_dim, activation='swish')
        self.ln1    = LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(num_classes)

    def build(self, input_shape):
        bridge_dim = input_shape[-1]
        self.dense1.build((None, bridge_dim))
        self.ln1.build((None, self.hidden_dim))
        self.dense2.build((None, self.hidden_dim))
        super().build(input_shape)

    def call(self, teacher_bridge_feat):
        pooled  = tf.reduce_mean(teacher_bridge_feat, axis=1)
        h       = self.ln1(self.dense1(pooled))
        logits  = self.dense2(h)
        return tf.nn.softmax(logits / self.soft_temp, axis=-1)


# ============================ 主蒸馏训练器 ============================
class MADSPDDistillerV4:
    def __init__(self,
                 teacher,
                 teacher_extractor,
                 student_train: tf.keras.Model,
                 student_infer: tf.keras.Model,
                 num_classes: int,
                 T_min=2.0, T_max=8.0,
                 lam_ce=0.28, lam_kd=0.32, lam_icd=0.25, lam_lstr=0.15,
                 label_smoothing=0.05,
                 save_dir: str = 'save1'):
        self.teacher     = teacher
        self.teacher_ext = teacher_extractor
        self.s_train     = student_train
        self.s_infer     = student_infer
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.lam = dict(ce=lam_ce, kd=lam_kd, icd=lam_icd, lstr=lam_lstr)
        self.save_dir    = save_dir

        self.sdat = SDAT(T_min=T_min, T_max=T_max)
        self.icd  = ICD(tau=0.05)
        self.lstr = LSTR(num_classes=num_classes, hidden_dim=64, soft_temp=2.0)

        # 余弦重启调度（PWS）
        self.optimizer = AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.005,
                first_decay_steps=16,
                t_mul=2.0, m_mul=0.9,
            ),
            weight_decay=4e-5,
            clipnorm=1.0,
        )
        self._epoch     = 0
        self._tot_epoch = 200

    def _warm_build_lstr(self, X_sample: np.ndarray):
        x2 = tf.constant(X_sample[:2], dtype=tf.float32)
        ext_out = self.teacher_ext(x2, training=False)
        if 'bridge' in ext_out:
            _ = self.lstr(ext_out['bridge'])
        else:
            _ = self.lstr(tf.zeros((2, 1, 128), dtype=tf.float32))

    def _weights(self):
        """PWS：余弦驱动的四路损失权重渐进调度"""
        p  = self._epoch / max(self._tot_epoch, 1)
        cp = float(np.cos(np.pi * p))
        w  = {
            'kd':   self.lam['kd']   * (1.0 + 0.50 * cp),
            'icd':  self.lam['icd']  * (1.0 + 0.40 * cp),
            'ce':   self.lam['ce']   * (1.0 + 0.50 * (1.0 - cp)),
            'lstr': self.lam['lstr'] * (1.0 + 0.30 * (1.0 - cp)),
        }
        s = sum(w.values()) + 1e-8
        return {k: v / s for k, v in w.items()}

    @tf.function
    def _train_step(self, x_batch, y_batch, w_ce, w_kd, w_icd, w_lstr):
        # 教师前向（冻结）
        t_final  = self.teacher(x_batch, training=False)
        t_logits = tf.math.log(tf.clip_by_value(t_final, 1e-10, 1.0))
        ext_out  = self.teacher_ext(x_batch, training=False)
        t_bridge = ext_out.get('bridge', None)    
        t_pre    = ext_out.get('pre_logit', None)  

        t_bridge_gap = (tf.reduce_mean(t_bridge, axis=1)
                        if t_bridge is not None else t_pre)

        trainable = (self.s_train.trainable_variables +
                     self.lstr.trainable_variables)

        with tf.GradientTape() as tape:
            s_outs       = self.s_train(x_batch, training=True)
            s_probs      = s_outs['probs']
            s_logits     = s_outs['logits']
            s_bridge_gap = s_outs['bridge_gap']   

            # L_CE with label smoothing
            num_cls = tf.cast(self.num_classes, tf.float32)
            y_onehot = tf.one_hot(y_batch, self.num_classes)
            smooth_labels = ((1.0 - self.label_smoothing) * y_onehot +
                             self.label_smoothing / num_cls)
            l_ce = tf.reduce_mean(tf.reduce_sum(
                -smooth_labels * tf.nn.log_softmax(s_logits, axis=-1), axis=-1))

            # L_KD with SDAT
            T  = tf.stop_gradient(self.sdat(t_logits))
            Te = T[:, None]
            soft_t = tf.nn.softmax(t_logits / Te, axis=-1)
            log_s  = tf.nn.log_softmax(s_logits / Te, axis=-1)
            kl_per = tf.reduce_sum(
                soft_t * (tf.math.log(soft_t + 1e-10) - log_s), axis=-1)
            l_kd = tf.reduce_mean(kl_per * (T ** 2))

            # L_ICD
            if t_bridge_gap is not None:
                l_icd = self.icd(t_bridge_gap, s_bridge_gap)
            else:
                l_icd = tf.constant(0.0)

            # L_LSTR
            if t_bridge is not None:
                lstr_soft = self.lstr(t_bridge)
                log_s2    = tf.nn.log_softmax(s_logits, axis=-1)
                l_lstr    = tf.reduce_mean(tf.reduce_sum(
                    lstr_soft * (tf.math.log(lstr_soft + 1e-10) - log_s2), axis=-1))
            else:
                l_lstr = tf.constant(0.0)

            l_total = w_ce * l_ce + w_kd * l_kd + w_icd * l_icd + w_lstr * l_lstr

        grads = tape.gradient(l_total, trainable)
        valid = [(g, v) for g, v in zip(grads, trainable) if g is not None]
        if valid:
            self.optimizer.apply_gradients(valid)

        return {'total': l_total, 'ce': l_ce, 'kd': l_kd,
                'icd': l_icd, 'lstr': l_lstr}

    def distill(self, X_tr, y_tr, X_val, y_val, X_test, y_test,
                epochs: int = 250, batch_size: int = 128):
        self._tot_epoch = epochs
        N       = len(X_tr)
        n_batch = N // batch_size

        print("  [Init] Warm-building LSTR...")
        self._warm_build_lstr(X_tr)
        lstr_p = sum(np.prod(v.shape) for v in self.lstr.trainable_variables)
        print(f"  [Init] LSTR built, params: {lstr_p:,}")

        best_val_acc = 0.0
        best_weights = None
        patience     = 100
        no_improve   = 0
        hist = {k: [] for k in ['total', 'ce', 'kd', 'icd', 'lstr', 'val_acc']}

        for ep in range(epochs):
            self._epoch = ep
            w    = self._weights()
            w_ce   = tf.constant(w['ce'],   dtype=tf.float32)
            w_kd   = tf.constant(w['kd'],   dtype=tf.float32)
            w_icd  = tf.constant(w['icd'],  dtype=tf.float32)
            w_lstr = tf.constant(w['lstr'], dtype=tf.float32)

            idx  = np.random.permutation(N)
            Xs, ys = X_tr[idx], y_tr[idx]

            ep_loss = {k: [] for k in ['total', 'ce', 'kd', 'icd', 'lstr']}
            for b in range(n_batch):
                xb = tf.constant(Xs[b * batch_size:(b + 1) * batch_size], dtype=tf.float32)
                yb = tf.constant(ys[b * batch_size:(b + 1) * batch_size], dtype=tf.int32)
                res = self._train_step(xb, yb, w_ce, w_kd, w_icd, w_lstr)
                for k in ep_loss:
                    ep_loss[k].append(float(res[k]))

            rem = N % batch_size
            if rem:
                xb = tf.constant(Xs[-rem:], dtype=tf.float32)
                yb = tf.constant(ys[-rem:], dtype=tf.int32)
                res = self._train_step(xb, yb, w_ce, w_kd, w_icd, w_lstr)
                for k in ep_loss:
                    ep_loss[k].append(float(res[k]))

            vp      = self.s_infer.predict(X_val, verbose=0, batch_size=256)
            val_acc = float(np.mean(np.argmax(vp, axis=1) == y_val))

            for k in ep_loss:
                hist[k].append(float(np.mean(ep_loss[k])))
            hist['val_acc'].append(val_acc)

            mark = ''
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.s_infer.get_weights()
                no_improve   = 0
                mark = '  ← ★ 最佳'
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"Ep {ep+1:03d}/{epochs} | "
                      f"L={hist['total'][-1]:.4f} "
                      f"CE={hist['ce'][-1]:.4f} "
                      f"KD={hist['kd'][-1]:.4f} "
                      f"ICD={hist['icd'][-1]:.4f} "
                      f"LSTR={hist['lstr'][-1]:.4f} | "
                      f"val={val_acc:.4f}{mark}")

            if no_improve >= patience:
                print(f"\n[早停] {patience} 轮无提升，Epoch {ep+1} 终止。")
                break

        if best_weights is not None:
            self.s_infer.set_weights(best_weights)
            print(f"\n[✓] 已恢复最佳权重  val_acc={best_val_acc:.4f}")

        # ===================== 保存训练 Loss 数据 =====================
        os.makedirs(self.save_dir, exist_ok=True)
        loss_df = pd.DataFrame({
            'epoch':      list(range(1, len(hist['total']) + 1)),
            'total_loss': hist['total'],
            'ce_loss':    hist['ce'],
            'kd_loss':    hist['kd'],
            'icd_loss':   hist['icd'],
            'lstr_loss':  hist['lstr'],
            'val_acc':    hist['val_acc'],
        })
        loss_csv_path = os.path.join(self.save_dir, 'distillation_loss_history.csv')
        loss_df.to_csv(loss_csv_path, index=False)

        self._final_eval(X_test, y_test)
        return hist

    def _final_eval(self, X_test: np.ndarray, y_test: np.ndarray):
        n_test = X_test.shape[0]
        os.makedirs(self.save_dir, exist_ok=True)

        _ = self.teacher.predict(X_test[:1], verbose=0, batch_size=1)   
        t0 = time.perf_counter()
        t_prob = self.teacher.predict(X_test, verbose=0, batch_size=256)
        t1 = time.perf_counter()

        teacher_total_s   = t1 - t0
        teacher_per_s     = teacher_total_s / n_test
        teacher_per_us    = teacher_per_s * 1e6

        t_pred = np.argmax(t_prob, axis=1)
        t_acc  = float(np.mean(t_pred == y_test))

        _ = self.s_infer.predict(X_test[:1], verbose=0, batch_size=1)   
        s0 = time.perf_counter()
        s_prob = self.s_infer.predict(X_test, verbose=0, batch_size=256)
        s1 = time.perf_counter()

        student_total_s   = s1 - s0
        student_per_s     = student_total_s / n_test
        student_per_us    = student_per_s * 1e6

        s_pred = np.argmax(s_prob, axis=1)
        s_acc  = float(np.mean(s_pred == y_test))

        print("  推理时间统计")
        print(f"\n  [Teacher]")
        print(f"    每样本推理时间   : {teacher_per_s:.6f} 秒  /  {teacher_per_us:.6f} μs")
        print(f"\n  [Student]")
        print(f"    每样本推理时间   : {student_per_s:.6f} 秒  /  {student_per_us:.6f} μs")


        # ── 教师模型详细评估指标 ──
        t_pred_classes = np.argmax(t_prob, axis=1)
        t_p, t_r, t_f, _ = precision_recall_fscore_support(
            y_test, t_pred_classes, average='weighted', zero_division=0
        )
        print("\n=== 教师模型评估结果 ===")
        print(f"加权精确率: {t_p:.4f}")
        print(f"加权召回率: {t_r:.4f}")
        print(f"加权F1分数: {t_f:.4f}")
        print("\n=== 教师详细分类报告 ===")
        print(classification_report(
            y_test, t_pred_classes,
            target_names=[f'Class_{i}' for i in range(self.num_classes)],
            zero_division=0
        ))

        # ── 学生模型详细评估指标 ──
        s_pred_classes = np.argmax(s_prob, axis=1)
        s_p, s_r, s_f, _ = precision_recall_fscore_support(
            y_test, s_pred_classes, average='weighted', zero_division=0
        )
        print("\n=== 学生模型评估结果 ===")
        print(f"加权精确率: {s_p:.4f}")
        print(f"加权召回率: {s_r:.4f}")
        print(f"加权F1分数: {s_f:.4f}")
        print("\n=== 学生详细分类报告 ===")
        print(classification_report(
            y_test, s_pred_classes,
            target_names=[f'Class_{i}' for i in range(self.num_classes)],
            zero_division=0
        ))


def load_enhanced_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test  = test_data.iloc[:, :-1].values
    y_test  = test_data.iloc[:, -1].values

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test  = np.where(np.isinf(X_test),  np.nan, X_test)

    max_v, min_v = 1e20, -1e20
    X_train = np.clip(X_train, min_v, max_v)
    X_test  = np.clip(X_test,  min_v, max_v)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test  = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test


def main():
    set_seeds(42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save1')
    os.makedirs(save_dir, exist_ok=True)

    X_tr_full, y_tr_full, X_te, y_te = load_enhanced_data(
        '../train.csv', '../test.csv')
    num_classes = len(np.unique(np.concatenate([y_tr_full, y_te])))

    SEQ_LEN = 2
    n_features = X_tr_full.shape[-1]
    chunk_dim  = int(np.ceil(n_features / SEQ_LEN))
    padded_dim = chunk_dim * SEQ_LEN
    pad_size   = padded_dim - n_features

    if pad_size > 0:
        X_tr_full = np.pad(X_tr_full, ((0, 0), (0, pad_size)),
                           mode='constant', constant_values=0)
        X_te      = np.pad(X_te,      ((0, 0), (0, pad_size)),
                           mode='constant', constant_values=0)

    X_tr_full = X_tr_full.reshape(-1, SEQ_LEN, chunk_dim).astype(np.float32)
    X_te      = X_te.reshape(-1, SEQ_LEN, chunk_dim).astype(np.float32)


    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2,
        random_state=42, stratify=y_tr_full)
    input_shape = (SEQ_LEN, chunk_dim)

    # 重建教师并加载权重（对应保存的 ec_net_model_weights5.h5）
    teacher = build_ec_net(input_shape, num_classes)
    _ = teacher(np.zeros((2, *input_shape), dtype=np.float32), training=False)
    teacher.load_weights('ec_net_model_weights5.h5')
    teacher.trainable = False
    t_params = teacher.count_params()
    print(f"      Teacher 参数量: {t_params:,}")
    t_prob = teacher.predict(X_te, verbose=0, batch_size=256)
    t_acc  = float(np.mean(np.argmax(t_prob, axis=1) == y_te))
    print(f"      Teacher 基准准确率: {t_acc:.4f}")

    # 构建极致轻量化学生模型
    student_train, student_infer = build_student(
        input_shape, num_classes, bridge_dim=32)
    _ = student_train(np.zeros((2, *input_shape), dtype=np.float32),
                      training=False)
    s_params = student_infer.count_params()
    print(f"      Student 参数量: {s_params:,}  "
          f"压缩率: {s_params/t_params:.2%}  "
          f"减少: {(1 - s_params/t_params)*100:.1f}%")

    # 构建教师中间层提取器
    teacher_ext = build_teacher_extractor(teacher)

    distiller = MADSPDDistillerV4(
        teacher=teacher,
        teacher_extractor=teacher_ext,
        student_train=student_train,
        student_infer=student_infer,
        num_classes=num_classes,
        T_min=2.0, T_max=8.0,
        lam_ce=0.25, lam_kd=0.25, lam_icd=0.25, lam_lstr=0.25,
        label_smoothing=0.05,
        save_dir=save_dir,
    )

    history = distiller.distill(
        X_train, y_train,
        X_val,   y_val,
        X_te,    y_te,
        epochs=150, batch_size=256,
    )

    student_weights_path = os.path.join(save_dir, 'student_mads_pd_v4_weights.h5')
    student_infer.save_weights(student_weights_path)
    return student_infer, teacher, history


if __name__ == '__main__':
    student, teacher, hist = main()