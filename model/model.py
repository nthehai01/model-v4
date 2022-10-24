import tensorflow as tf
from tensorflow.keras.layers import Dense

from model.notebook_transformer import NotebookTransformer

class Model(tf.keras.Model):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        super(Model, self).__init__()
        self.notebook_transformer = NotebookTransformer(d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers)
        self.top = Dense(2, activation='sigmoid')  # (rank, {cell_type}_rank)


    def call(self, x, additional_features, count_by_type, cell_count, is_training, mask=None):
        out = self.notebook_transformer(x, is_training, mask)  # (..., max_cells, d_model)
        out = tf.concat([out, additional_features], axis=-1)   # (..., max_cells, d_model+6)

        out = self.top(out)  # (..., max_cells, 2)

        counts = tf.concat([count_by_type, cell_count], axis=-1)  # (..., max_cells, 2)
        out = tf.multiply(out, counts)  # (..., max_cells, 2)

        out = tf.math.round(out)
        
        return out
