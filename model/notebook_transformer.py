import tensorflow as tf
from tensorflow.keras.layers import Layer

from model.transformer_utils.encoder_block import EncoderBlock

class NotebookTransformer(Layer):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        super(NotebookTransformer, self).__init__()
        self.encoder_layers = [EncoderBlock(d_model, n_heads, dropout, eps, d_ff, ff_activation) for _ in range(n_layers)]


    def call(self, x, is_training, mask=None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, is_training, mask)
        return x