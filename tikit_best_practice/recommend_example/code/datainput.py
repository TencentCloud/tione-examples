from re import T
import numpy as np
import tensorflow as tf


class DataInput:
    """
     old:
     train input : [bs,slots]->[bs,slots,1]->sparse[bs,slots,*totalkeys]->[bs,slots,dim]
     infer input : [bs,slots]->

     for example:
     train : dense tensor-> sparse tensor tf.SparseTensor: [0,0,A],[0,0,B],[0,1,D],[1,0,B],[1,1,F]...
     infer : tf.SparseTensor -> 3 placeholders -> tf.SparseTensor

     new:
     train input : fea_64_1:sparse[bs,1],fea_64_2:sparse[bs,1]->dim-64:sparse[bs,slots,*totalkeys]->[bs,slots,dim]
     infer input : [bs,slots]->

     for example:
     train : sparse tensor tf.SparseTensor: fea_64_1:[0,A],[0,B],[1,C],[2,D]... fea_64_1:[0,E],[1,F],[2,G],[2,H]...
        -> sparse tensor tf.SparseTensor: [0,0,0],[0,0,1],[0,1,0],[1,0,0]...
     infer : tf.SparseTensor -> 3 placeholders -> tf.SparseTensor
     infer : train-> is_batch=True, infer-> is_batch=False

    """

    def __init__(self, batch_size, origin_critieo_benchmark=True):
        super(DataInput, self).__init__()
        self.batch_size = batch_size
        self.origin_critieo_benchmark = origin_critieo_benchmark
        if self.origin_critieo_benchmark == True:
            self.epoch = 1
            self.dense_num = 13
            self.sparse_num = 26
            self.feature_num = self.dense_num + self.sparse_num
        else:
            print("Not supported!")

    # dataset_input for critieo_benchmark mode
    def dataset_input(self, data_files, infer=False):
        dataset = tf.data.TextLineDataset(data_files). \
        repeat(self.epoch). \
        batch(self.batch_size, drop_remainder=True). \
        map(lambda record_batch: self.parser_fn_dataset(record_batch, infer), num_parallel_calls=tf.data.experimental.AUTOTUNE). \
        prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def parser_fn_dataset(self, record_batch, infer=False):
        # must return dict features, labels

      def _to_tensors(record_batch):
        '''This Dataset parser is for Criteo official format.'''
        decode_batch = [x.decode('utf-8') for x in record_batch]
        str_data = np.char.split(decode_batch, sep='\t')

        labels = np.vectorize(lambda x: x[0])(str_data).astype(np.float32).reshape(
            (-1, 1))

        ft_dense = np.asarray([x[1:14] for x in str_data]).reshape([-1])
        ft_dense = np.where(ft_dense == '', '0', ft_dense)
        ft_dense = ft_dense.astype(dtype=np.int64).reshape(
            [self.batch_size, self.dense_num])
        ft_dense = ft_dense + np.array(
            [(i + 1) * 0xFFFFFFFF for i in range(self.dense_num)])
        ft_dense = ft_dense.astype(dtype=np.int64).reshape(
            [self.batch_size, self.dense_num])

        ft_sparse = np.asarray([x[14:] for x in str_data]).reshape([-1])
        ft_sparse = np.where(ft_sparse == '', '0xFFFFFFFF', ft_sparse)
        ft_sparse = np.asarray([int(x, 16) for x in ft_sparse],
                                dtype=np.int64).reshape([-1, self.sparse_num])
        ft_sparse = np.concatenate([ft_dense, ft_sparse], axis=1)

       
        ft_sparse_val = ft_sparse.astype(dtype=np.int64).reshape([-1])
        ft_sparse_idx = [[num // self.feature_num, num % self.feature_num, 0] for num in range(self.feature_num * self.batch_size)]
        

        return ft_sparse_idx, ft_sparse_val, labels
    
      ft_sparse_idx, ft_sparse_val, labels = tf.compat.v1.py_func(_to_tensors,
                                                                  stateful=False,
                                                                  inp=[record_batch],
                                                                  Tout=[tf.int64, tf.int64, tf.float32])
      labels = tf.reshape(labels, shape=[-1, 1]) 
      ft_sparse_val = tf.reshape(ft_sparse_val, shape=[-1])
      ft_sparse_idx = tf.reshape(ft_sparse_idx, shape=[-1, 3])
      dense_shape = [self.batch_size, self.feature_num, 1]
      key_sparse_tensor = tf.sparse.SparseTensor(ft_sparse_idx, ft_sparse_val, dense_shape) 
      output = {'fea1': key_sparse_tensor}  # a dict is used for estimator.head 
      return output, labels


