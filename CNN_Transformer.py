# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import regularizers, Input
from keras.layers import concatenate
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCR_GPU_ALLOW_GROWTH"] = "true"  # Attempt to allocate GPU memory dynamically

import tensorflow as tf
from tensorflow.keras import layers, models

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.norm = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.conv = layers.Conv1D(dim, 3, activation='gelu', padding='same')
        # self.pool = layers.MaxPooling1D(2)

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = self.attn(x, x)
        print("x", x.shape)
        x = self.conv(x)
        print("x", x.shape)
        # x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads
        })
        return config

from FFT import FourierUnit

class AdaptiveFusionBlock(layers.Layer):
    def __init__(self, dim, **kwargs):
        super(AdaptiveFusionBlock, self).__init__(**kwargs)
        self.swin_transformer = SwinTransformerBlock(dim, num_heads=4)
        self.conv = layers.Conv1D(dim, kernel_size=3, padding="same", activation="relu")
        self.concat = layers.Concatenate()
        self.inputs_proj = layers.Conv1D(filters=dim*2, kernel_size=1, padding='same')

    def call(self, inputs, training=False):
        swin = self.swin_transformer(inputs, training=training)
        print("swin", swin.shape)
        conv = self.conv(inputs)
        print("conv", conv.shape)
        concat = self.concat([swin, conv])
        print("concat", concat.shape)
        inputs_proj = self.inputs_proj(inputs)
        print("inputs_proj", inputs_proj.shape)
        return layers.Add()([inputs_proj, concat])

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.swin_transformer.dim
        })
        return config

class CATM(layers.Layer):
    def __init__(self, dim, **kwargs):
        super(CATM, self).__init__(**kwargs)
        self.dim = dim
        self.swin_transformer = SwinTransformerBlock(dim, num_heads=4)
        self.concat = layers.Concatenate()
        self.conv = layers.Conv1D(dim, kernel_size=1, activation="relu")

        # Use Conv1D instead of Dense to generate weights
        self.weight_conv_skip = layers.Conv1D(1, kernel_size=1, activation='sigmoid')
        self.weight_conv_decoder = layers.Conv1D(1, kernel_size=1, activation='sigmoid')

    def call(self, inputs, training=False):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("Expected input to be a list or tuple of two elements: [x_skip, x_decoder]")
        x_skip, x_decoder = inputs

        # Process x_decoder through SwinTransformer
        swin = self.swin_transformer(x_decoder, training=training)

        # Generate weighting coefficients (reduce channel to 1)
        weight_skip = self.weight_conv_skip(x_skip)
        weight_decoder = self.weight_conv_decoder(swin)

        # Apply weights
        weighted_skip = x_skip * weight_skip
        weighted_decoder = swin * weight_decoder

        # Concatenate weighted features
        concat = self.concat([weighted_skip, weighted_decoder])

        # Fuse with final 1x1 convolution
        return self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim
        })
        return config

from mamba1 import Mamba, MambaBlock
from HAAM import HAAM
from FFTBlock import FFTNetBlock

def build_model(input_shape=(1440, 3), output_classes=2):
    inputs = layers.Input(shape=input_shape)

    # === CNN Encoder (ResNet-like) ===
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    f1 = layers.MaxPooling1D(pool_size=2)(x)  # Level 1
    print("f1", f1.shape)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(f1)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    f2 = layers.MaxPooling1D(pool_size=2)(x)  # Level 2
    print("f2", f2.shape)

    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(f2)
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    f3 = layers.MaxPooling1D(pool_size=2)(x)  # Level 3
    print("f3", f3.shape)

    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(f3)
    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    f4 = layers.MaxPooling1D(pool_size=2)(x)  # Level 4
    print("f4", f4.shape)

    # === Transformer Encoder ===
    g1 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(f1)
    g1 = Mamba(seq_len=1440, d_model=512, state_size=8)(g1)
    print("g1", g1.shape)
    g11 = layers.MaxPooling1D(pool_size=2)(g1)

    g2 = Mamba(seq_len=720, d_model=256, state_size=8)(g11)
    print("g2", g2.shape)
    g22 = layers.MaxPooling1D(pool_size=2)(g2)

    g3 = Mamba(seq_len=360, d_model=128, state_size=8)(g22)
    print("g3", g3.shape)
    g33 = layers.MaxPooling1D(pool_size=2)(g3)

    g4 = Mamba(seq_len=180, d_model=64, state_size=8)(g33)
    print("g4", g4.shape)
    g44 = layers.MaxPooling1D(pool_size=2)(g4)

    # === Multi-scale Feature Complementary Module (like CATM) ===
    m1 = CATM(64)([g1, f1])
    print("m1", m1.shape)
    m2 = CATM(128)([g2, f2])
    print("m2", m2.shape)
    m3 = CATM(256)([g3, f3])
    print("m3", m3.shape)
    m4 = CATM(512)([g4, f4])
    print("m4", m4.shape)

    # === Transformer Decoder ===
    u1 = AdaptiveFusionBlock(64)(m4)
    u1 = concatenate([u1, m4])
    u1 = layers.UpSampling1D(size=2)(u1)
    print("u1", u1.shape)

    u2 = AdaptiveFusionBlock(256)(u1)
    u2 = concatenate([u2, m3])
    u2 = layers.UpSampling1D(size=2)(u2)
    print("u2", u2.shape)

    u3 = AdaptiveFusionBlock(128)(u2)
    u3 = concatenate([u3, m2])
    u3 = layers.UpSampling1D(size=2)(u3)
    print("u3", u3.shape)

    u4 = AdaptiveFusionBlock(32)(u3)
    u4 = concatenate([u4, m1])
    u4 = layers.UpSampling1D(size=2)(u4)

    # Final output layer
    outputs = layers.Conv1D(output_classes, kernel_size=1, activation='softmax')(u4)

    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
