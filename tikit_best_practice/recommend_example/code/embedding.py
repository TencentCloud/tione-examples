import tensorflow as tf
import layers


class EmbeddingModel(tf.keras.Model):
    def __init__(self, num_buckets, embedding_dim=16):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.emb_layer = layers.HashEmbedding(self.num_buckets, self.embedding_dim, name="emb_layer")

    def call(self, inputs, training=False, mask=None):
        outputs = self.emb_layer(inputs)
        return outputs
