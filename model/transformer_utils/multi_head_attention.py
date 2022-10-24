import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads = 6):
        """
        Args:
            d_model (int): Dimensionality of the feature embedding.
            num_heads (int): the number of heads.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_linear = Dense(d_model)
        self.k_linear = Dense(d_model)
        self.v_linear = Dense(d_model)
        self.out_linear = Dense(d_model)


    def split_head(self, x):
        """
        Split the last dimension into (num_heads, d_head).
        
        Args:
            x (tensor): input with shape (..., seqlen, d_model)
        Returns:
            tensor: split tensor with shape (..., num_heads, seqlen, d_head)
        """

        d_head = self.d_model // self.num_heads
        seqlen = tf.shape(x)[1]

        x = tf.reshape(x, (-1, seqlen, self.num_heads, d_head))

        return tf.transpose(x, perm=[0, 2, 1, 3])


    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """ 
        Scaled dot-product attention.

        Args:
            q (tensor): query with shape (..., q_length, d_model)
            k (tensor): key with shape (..., k_length, d_model)
            v (tensor): value with shape (..., v_length, d_model)
            k_length = v_length
        Returns:
            attention (tensor): self attention with shape (..., q_length, k_length)
        """
        
        assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Embedding dimensions of q, k, v aren't all the same"
        assert k.shape[-2] == v.shape[-2], "Key and value lengths aren't the same"

        depth = tf.shape(q)[-1]
        depth = tf.cast(depth, tf.float32)

        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(depth)  # shape (..., q_length, k_length)

        if mask:
            attention_scores += (mask * -1e30)

        attention = tf.nn.softmax(attention_scores, axis=-1)
        attention = tf.matmul(attention, v)  # shape (..., q_length, d_v)

        return attention


    def call(self, q, k, v, mask=None):
        """
        Multi-head Attention.

        Args:
            q (tensor): query with shape (..., q_length, d_model)
            k (tensor): key with shape (..., k_length, d_model)
            v (tensor): value with shape (..., v_length, d_model)
            k_length = v_length
        Returns:
            attention_matrix (tensor): self attention with shape (..., q_length, k_length)
        """

        q_length = tf.shape(q)[1]

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q_h = self.split_head(q)  # shape (..., num_heads, q_length, d_head)
        k_h = self.split_head(k)  # shape (..., num_heads, k_length, d_head)
        v_h = self.split_head(v)  # shape (..., num_heads, v_length, d_head)

        attention_matrix = self.scaled_dot_product_attention(q_h, k_h, v_h, mask)  # shape (..., num_heads, q_length, d_head)

        attention_matrix = tf.transpose(attention_matrix, perm=[0, 2, 1, 3])  # shape (..., q_length, num_heads, d_head)
        attention_matrix = tf.reshape(attention_matrix, (-1, q_length, self.d_model))  # shape (..., q_length, d_model)
        
        attention_matrix = self.out_linear(attention_matrix)

        return attention_matrix
        