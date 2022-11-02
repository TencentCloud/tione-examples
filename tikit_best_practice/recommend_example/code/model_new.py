import math
import tensorflow as tf
import re

from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator.canned import optimizers

from embedding import EmbeddingModel

# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05

_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


def _add_hidden_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)


def _get_previous_name_scope():
  current_name_scope = ops.get_name_scope()
  return current_name_scope.rsplit('/', 1)[0] + '/'


def _linear_learning_rate(num_linear_feature_columns):
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(_LINEAR_LEARNING_RATE, default_learning_rate)

def convert_tensor_to_sparse_tensor(features):
    ret = {}
    for i in features:
        if i.startswith('_c_'):
            if i.startswith('_c_indices_'):
                i = i[len('_c_indices_'):]
                ret[i] = tf.SparseTensor(features['_c_indices_' + i],
                                         features['_c_values_' + i],
                                         tf.cast(features['_c_dense_shape_' + i], tf.int64))
        else:
            ret[i] = features[i]
    return ret


class NewDeepFM:
    def __init__(self,
                 config=None,
                 origin_critieo_benchmark=True):
        self.batch_size = 512
        self.config = config
        self.origin_critieo_benchmark = origin_critieo_benchmark
        self.dense_num = 13
        self.sparse_num = 26
        self.feature_num = self.dense_num + self.sparse_num
        self.embedding_size = 10
        self.dense_weights = []

    def fc(self, inputs, w_shape, b_shape, name):
        weight = tf.compat.v1.get_variable(name="%s_weights" % name,
                                       initializer=tf.random.normal(w_shape,
                                                                    mean=0.0,
                                                                    stddev=0.1))
        bias = tf.compat.v1.get_variable(name="%s_bias" % name,
                                        initializer=tf.random.normal(b_shape,
                                                                    mean=0.0,
                                                                    stddev=0.1))
        self.dense_weights.append(weight)
        self.dense_weights.append(bias)

        return tf.compat.v1.nn.xw_plus_b(inputs, weight, bias)    
       

    def model_fn(self, features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = convert_tensor_to_sparse_tensor(features)
        
        # wide lookup embedding
        emb_parent_scope = 'wide_emb'
        with variable_scope.variable_scope(emb_parent_scope) as scope:
            emb_absolute_scope = scope.name

            if self.origin_critieo_benchmark:
                wide_emb_model = EmbeddingModel(params['emb_size'], 1)  
                sp_test_values = tf.mod(features['fea1'].values, params['emb_size'])
                fea = tf.SparseTensor(indices=features['fea1'].indices,
                                    values=sp_test_values, dense_shape=features['fea1'].dense_shape)
                wide_emb_outputs = wide_emb_model(fea)
            else:
                print("only support origin_critieo_benchmark")                                                
            
            #wide_emb_outputs = tf.Print(wide_emb_outputs, [tf.shape(wide_emb_outputs)], 'emb output shape:')
            wide_emb_outputs = tf.reshape(wide_emb_outputs,shape=[-1, 39 * 1])
            #wide_emb_outputs = tf.Print(wide_emb_outputs, [tf.shape(wide_emb_outputs)], 'after reshape emb output shape:')
            
        wide_bias = tf.compat.v1.get_variable(name="wide_bias",initializer=tf.random.normal([1], mean=0.0, stddev=0.1))
        self.dense_weights.append(wide_bias)
        wide = tf.reshape(tf.reshape(tf.reduce_sum(input_tensor=wide_emb_outputs, axis=1),
                          shape=[self.batch_size, 1]) + wide_bias,
                          shape=[self.batch_size, 1])


        # deep lookup embedding
        emb_parent_scope = 'deep_emb'
        with variable_scope.variable_scope(emb_parent_scope) as scope:
            emb_absolute_scope = scope.name
            
            if self.origin_critieo_benchmark:
                deep_emb_model = EmbeddingModel(params['emb_size'], 10)  # TODO out of index
                sp_test_values = tf.mod(features['fea1'].values, params['emb_size'])
                fea = tf.SparseTensor(indices=features['fea1'].indices,
                                    values=sp_test_values, dense_shape=features['fea1'].dense_shape)
                deep_emb_outputs = deep_emb_model(fea)
            else:
                print("only support origin_critieo_benchmark")                                                

            
            #deep_emb_outputs = tf.Print(deep_emb_outputs, [tf.shape(deep_emb_outputs)], 'emb output shape:')
            deep_emb_outputs = tf.reshape(deep_emb_outputs,shape=[-1, 39 * 10])
            #deep_emb_outputs = tf.Print(deep_emb_outputs, [tf.shape(deep_emb_outputs)], 'after reshape emb output shape:')
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            _rate = 0.2
        elif mode == tf.estimator.ModeKeys.EVAL:
            _rate = 0.0
        else:
            _rate = 0.0
            
        deep = self.fc(deep_emb_outputs,[self.feature_num * self.embedding_size, 400], [400],"fc_0")
        deep = tf.nn.dropout(deep, rate=_rate)
        deep = tf.nn.relu(deep)
        deep = self.fc(deep, [400, 400], [400], "fc_1")
        deep = tf.nn.dropout(deep, rate=_rate)
        deep = tf.nn.relu(deep)
        deep = self.fc(deep, [400, 1], [1], "fc_2")          
            
        fm_parent_scope = 'fm'
        with variable_scope.variable_scope(fm_parent_scope) as scope:
            x = tf.reshape(deep_emb_outputs, 
                           shape=[-1, self.feature_num, self.embedding_size])
            fm = tf.reshape(0.5 * tf.reduce_sum(
                input_tensor=(tf.pow(tf.reduce_sum(input_tensor=x, axis=1), 2) -
                              tf.reduce_sum(input_tensor=tf.pow(x, 2), axis=[1])),
                axis=[1],
                keepdims=True),
                            shape=[self.batch_size, 1])

        logits = deep + wide + fm
        logits = tf.Print(logits, [tf.shape(logits)], 'logits shape:')
        
        # calc loss
        predict = tf.nn.sigmoid(logits)

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=tf.reshape(logits, shape=[-1]),
                                  labels=tf.reshape(labels, shape=[-1])))
        else:
            loss = None

        if mode == tf.estimator.ModeKeys.EVAL:
            auc = tf.metrics.auc(labels=labels, predictions=predict, name='auc_op', summation_method='careful_interpolation')
            metrics = {'auc': auc}
        else:
            metrics = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
            'probabilities': predict,
            'logits': logits,
            }
        else:
            predictions = None   
                       
        # train op -> optimizer.minimize
        if mode ==  tf.estimator.ModeKeys.TRAIN:
            opt = tf.compat.v1.train.AdamOptimizer(0.001 * self.batch_size / 2048)
            train_op = opt.minimize(loss, global_step=training_util.get_global_step())
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=metrics)
                
