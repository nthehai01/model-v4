import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

from model.transformer_utils.multi_head_attention import MultiHeadAttention

class EncoderBlock(Layer):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNormalization(epsilon=eps)
        self.norm2 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.feed_forward = tf.keras.Sequential([
            Dense(d_ff, activation=ff_activation),
            Dense(self.d_model)
        ])


    def call(self, x, is_training, mask=None):
        # Multi-head attention
        mha_output = self.mha(x, x, x, mask)
        mha_output = self.dropout1(mha_output, training=is_training)

        # Add & Norm
        x = x + mha_output
        x = self.norm1(x)

        # Feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output, training=is_training)

        # Add & Norm
        x = x + ff_output
        x = self.norm2(x)
        
        return x
