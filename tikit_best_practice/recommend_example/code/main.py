import argparse
import string
import numpy as np
import tensorflow as tf
import os
import json
import socket

from model_new import NewDeepFM
from datainput import DataInput
from datetime import datetime

tf.compat.v1.disable_eager_execution()

def build_example_placeholder_criteo():
    fea = 'fea1'
    output = {}
    output['_c_indices_' + fea] = tf.compat.v1.placeholder(tf.int64, shape=(None, 3))
    output['_c_values_' + fea] = tf.compat.v1.placeholder(tf.int64, shape=(None,))
    output['_c_dense_shape_' + fea] = tf.compat.v1.placeholder(tf.int64, shape=(3,))

    return output



def create_parser():
    """Initialize command line parser using arparse.

    Returns:
      An argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--amp',
        help='Attempt automatic mixed precision conversion',
        default=False,
        action='store_true')
    parser.add_argument(
        '--hvd',
        help='Use Horovod',
        action='store_true',
        default=False)
    parser.add_argument(
        '--training_set_size',
        help='Number of samples in the training set',
        default=1024,
        type=int)
    parser.add_argument(
        '--global_batch_size',
        help='Total training batch size',
        default=512,
        type=int)
    parser.add_argument(
        '--linear_learning_rate',
        help='Learning rate for linear model',
        type=float,
        default=0.2)
    parser.add_argument(
        '--linear_l1_regularization',
        help='L1 regularization for linear model',
        type=float,
        default=0.0)
    parser.add_argument(
        '--linear_l2_regularization',
        help='L2 regularization for linear model',
        type=float,
        default=0.0)
    parser.add_argument(
        '--deep_learning_rate',
        help='Learning rate for deep model',
        type=float,
        default=1.0)
    parser.add_argument(
        '--deep_warmup_epochs',
        help='Number of epochs for deep LR warmup',
        type=float,
        default=0)
    parser.add_argument(
        '--origin_critieo_benchmark', # for testorigin critieo dataset
        default=True,
        action='store_true')         
    parser.add_argument(
        '--finetune', 
        default=False,
        action='store_true')
    parser.add_argument(
        '--finetune_path',
        type=str,
        default="/opt/ml/output/ckpt/model.ckpt-1")     
    parser.add_argument(
        '--model_dir',
        type=str,
        default="/opt/ml/model/ckpt/")     
    parser.add_argument(
        '--train_file',
        type=str,
        default="/opt/ml/input/data/criteo/train_new.txt")    
    parser.add_argument(
        '--eval_file',
        type=str,
        default="/opt/ml/input/data/criteo/val.txt")    
    parser.add_argument(
        '--embedding_size',
        help='embedding size',
        type=int,
        default=4000000)
    parser.add_argument(
        '--hiddens',
        nargs='+',
        type=int,
        default=[1024, 1024])

    return parser


def main(FLAGS):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    steps_per_epoch = FLAGS.training_set_size / FLAGS.global_batch_size

    def learning_rate_scheduler(lr_init, warmup_steps, global_step):
        warmup_lr = (lr_init * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

        return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr_init)

    def wide_optimizer():
        opt = tf.compat.v1.train.FtrlOptimizer(
            learning_rate=FLAGS.linear_learning_rate,
            l1_regularization_strength=FLAGS.linear_l1_regularization,
            l2_regularization_strength=FLAGS.linear_l2_regularization)
        if FLAGS.amp:
            loss_scale = tf.train.experimental.DynamicLossScale()
            opt = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
        return opt

    def deep_optimizer():
        # learning rate here
        with tf.device("/cpu:0"):
            learning_rate_fn = learning_rate_scheduler(
                lr_init=FLAGS.deep_learning_rate,
                warmup_steps=int(steps_per_epoch * FLAGS.deep_warmup_epochs),
                global_step=tf.compat.v1.train.get_global_step()
            )
        opt = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=learning_rate_fn,
            initial_accumulator_value=0.1,
            use_locking=False)
        if FLAGS.amp:
            loss_scale = tf.train.experimental.DynamicLossScale()
            opt = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
        return opt

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=400, keep_checkpoint_max=20, log_step_count_steps=100)



    ds = DataInput(FLAGS.global_batch_size,FLAGS.origin_critieo_benchmark)


    
    model = NewDeepFM(config=None,  origin_critieo_benchmark=FLAGS.origin_critieo_benchmark)

  

    train_hooks = [tf.estimator.ProfilerHook(save_steps=5000, output_dir=FLAGS.model_dir)]
    
    if FLAGS.finetune:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=FLAGS.finetune_path)
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params={'emb_size': FLAGS.embedding_size},
            warm_start_from=ws
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params={'emb_size': FLAGS.embedding_size},
        )
    
    
    if FLAGS.origin_critieo_benchmark == True:

        serving_input_tensors = build_example_placeholder_criteo()
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(serving_input_tensors)
        exporter = tf.estimator.LatestExporter('saved_model', serving_input_receiver_fn, exports_to_keep=1)

        train_files = [FLAGS.train_file]
        test_files = [FLAGS.eval_file]
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: ds.dataset_input(train_files), max_steps=None, hooks=train_hooks)
        eval_spec = tf.estimator.EvalSpec(name=datetime.utcnow().strftime("%H%M%S"),
                                          input_fn=lambda: ds.dataset_input(test_files), exporters=[exporter], steps=None, start_delay_secs=0, throttle_secs=10)
    else: 
        # fusion or origin
        print("only support origin_critieo_benchmark")
        return 
    
    
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)
    
def modify_tf_config():
    tf_config = json.loads(os.getenv("TF_CONFIG"))

    chiefs = []
    if len(tf_config["cluster"]["worker"]) <= 1:
        chiefs = tf_config["cluster"]["worker"]
        del tf_config["cluster"]["worker"]
    else:
        chief = tf_config["cluster"]["worker"].pop(0)
        chiefs = [chief]
        
    tf_config["cluster"]["chief"] = chiefs
    
    if tf_config["task"]["type"] == "worker" and tf_config["task"]["index"] == 0:
        if os.getenv("CHIEF_OR_EVALUATOR"):
            if os.getenv("CHIEF_OR_EVALUATOR") == "1":
                tf_config["task"]["type"] = "chief"
            elif os.getenv("CHIEF_OR_EVALUATOR") == "2":
                tf_config["task"]["type"] = "evaluator"
            else:
                raise Exception("Invalid CHIEF_OR_EVALUATOR value")
        else:
            raise Exception("The CHIEF_OR_EVALUATOR environment variable required")

    os.environ["TF_CONFIG"] = json.dumps(tf_config)

if __name__ == '__main__':
    modify_tf_config()
    FLAGS = create_parser().parse_args()
    main(FLAGS)
