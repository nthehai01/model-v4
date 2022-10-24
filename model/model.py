import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

from model.notebook_transformer import NotebookTransformer

class Model(tf.keras.Model):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        super(Model, self).__init__()
        self.notebook_transformer = NotebookTransformer(d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers)
        self.top = Dense(1, activation='sigmoid')


    def call(self, x, is_training, mask=None):
        max_cells = tf.shape(x)[1]

        x = tf.cast(x, tf.float32)

        out = self.notebook_transformer(x, is_training, mask)
        out = self.top(out)
        out = tf.reshape(out, (max_cells,))
        
        return out
