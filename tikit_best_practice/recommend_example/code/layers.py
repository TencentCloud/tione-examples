import tensorflow as tf


class HashEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_buckets, embedding_dim, trainable=True, **kwargs):
        super(HashEmbedding, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.trainable = trainable

    def build(self, input_shape):
        shape = tf.TensorShape([self.num_buckets, self.embedding_dim])
        initializer = tf.random_uniform_initializer()
        self.embedding = tf.Variable(
            initial_value=initializer(shape=shape, dtype=tf.float32),
            shape=shape, dtype=tf.float32, trainable=self.trainable,
        )

    def call(self, inputs, **kwargs):
        outputs = tf.nn.safe_embedding_lookup_sparse(self.embedding, sparse_ids=inputs, sparse_weights=None)
        return outputs
