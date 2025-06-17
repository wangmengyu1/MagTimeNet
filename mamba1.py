# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.eps = eps
        self.weight = None  # Lazy initialization

    def build(self, input_shape):
        d_model = input_shape[-1]  # Feature dimension
        self.weight = self.add_weight(shape=(d_model,), initializer="ones", trainable=True)

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps) * self.weight

    def get_config(self):
        config = super(RMSNorm, self).get_config()
        config.update({"eps": self.eps})
        return config


class S6(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(S6, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        # Using Conv1D instead of Dense
        self.fc1 = layers.Conv1D(filters=d_model, kernel_size=3, padding="same")
        self.fc2 = layers.Conv1D(filters=state_size, kernel_size=3, padding="same")
        self.fc3 = layers.Conv1D(filters=state_size, kernel_size=3, padding="same")
        self.A = self.add_weight(shape=(d_model, state_size), initializer="glorot_uniform", trainable=True)

    def call(self, x):
        B = self.fc2(x)
        C = self.fc3(x)
        delta = tf.nn.softplus(self.fc1(x))
        dB = tf.einsum("bld,bln->bldn", delta, B)
        dA = tf.exp(tf.einsum("bld,dn->bldn", delta, self.A))
        h = tf.zeros_like(dB)
        h = tf.einsum('bldn,bldn->bldn', dA, h) + tf.expand_dims(x, -1) * dB
        y = tf.einsum('bln,bldn->bld', C, h)
        return y

    def get_config(self):
        config = super(S6, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "state_size": self.state_size,
        })
        return config


class MambaBlock(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.norm = RMSNorm()
        self.inp_proj = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same")
        self.out_proj = layers.Conv1D(filters=d_model, kernel_size=3, padding="same")
        self.conv = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same", activation="swish")
        self.conv_linear = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same")
        self.s6 = S6(seq_len, 2 * d_model, state_size)
        self.residual = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same", activation="swish")

    def call(self, x):
        x = self.norm(x)
        print("x", x.shape)
        x_proj = self.inp_proj(x)
        print("x_proj", x_proj.shape)
        x_conv = self.conv(x_proj)
        print("x_conv", x_conv.shape)
        x_conv_out = self.conv_linear(x_conv)
        print("x_conv_out", x_conv_out.shape)
        x_ssm = self.s6(x_conv_out)
        print("x_ssm", x_ssm.shape)
        x_residual = self.residual(x)
        print("x_residual", x_residual.shape)
        x_combined = x_ssm * x_residual
        print("x_combined", x_combined.shape)
        return self.out_proj(x_combined)

    def get_config(self):
        config = super(MambaBlock, self).get_config()
        config.update({
            "seq_len": self.s6.seq_len,
            "d_model": self.s6.d_model,
            "state_size": self.s6.state_size,
        })
        return config


class Mamba(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(Mamba, self).__init__(**kwargs)
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size)
        self.conv1d = layers.Conv1D(filters=d_model, kernel_size=1)

    def call(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return self.conv1d(x)

    def get_config(self):
        config = super(Mamba, self).get_config()
        config.update({
            "seq_len": self.mamba_block1.s6.seq_len,
            "d_model": self.mamba_block1.s6.d_model,
            "state_size": self.mamba_block1.s6.state_size,
        })
        return config


if __name__ == "__main__":
    seq_len = 1440  # Set sequence length
    d_model = 3     # Feature dimension
    state_size = 16 # S6 state size
    batch_size = 2  # Test batch size

    # Create Mamba model
    model = Mamba(seq_len, d_model, state_size)

    # Generate random input (batch_size, seq_len, d_model)
    x_test = tf.random.normal((batch_size, seq_len, d_model))

    # Run model
    y_test = model(x_test)

    # Print input and output shapes
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_test.shape}")  # Expected shape: (batch_size, seq_len, d_model)
