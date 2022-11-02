import numpy as np
import pandas as pd
import tensorflow as tf
import tiinfer
import os
import logging
from typing import Dict
from sklearn.metrics import roc_auc_score

MODEL_PATH_PREFIX = "model"

# 找到第一个包含savedmodel.pb的目录
def find_saved_model_path(prefix: str, ext: str)->str:
    from pathlib import Path
    for root, dirs, files in os.walk(prefix):
        for file in files:
            if Path(file).suffix == ext:
                return root
        for dir in dirs:
            new_dir = os.path.join(root, dir)
            r = find_saved_model_path(new_dir, ext)
            if r is not None:
                return r
    return None

class RecommendModel(tiinfer.Model):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.model_dir = model_dir
        self.model_name = "recommend"
        self.ready = False
        self.device_gpu = False
    
    def load(self) -> bool:
        try:
            self.sess = tf.Session(graph=tf.Graph())
            model_dir = find_saved_model_path(os.path.join(self.model_dir, MODEL_PATH_PREFIX), '.pb') 
            tf.saved_model.loader.load(
                self.sess,
                [tf.saved_model.tag_constants.SERVING],
                model_dir,
            )
            self.device_gpu = tf.test.is_gpu_available()
            self.feature_num = 39
            logging.info("load model with device_gpu [%s]", self.device_gpu)
        except Exception as e:
            logging.error("load model failed: %s", e)
            self.ready = False
            return self.ready
        self.ready = True
        return self.ready

    def preprocess(self, request: Dict) -> Dict:
        indexes = np.array(request['indexes'])
        values = np.array(request['values'])
        others = np.array([request['samples'], self.feature_num, 1])
        print('indexes shape:', indexes.shape)
        print('values shape:', values.shape)
        print('others:', others)

        return {
            'labels': request['labels'],
            'indexes': indexes,
            'values': values,
            'others': others,
        }

    def predict(self, request: Dict) -> Dict:
        pred = self.sess.run(
            'Sigmoid:0',
            feed_dict= {
                'Placeholder:0': request['indexes'],
                'Placeholder_1:0': request['values'],
                'Placeholder_2:0': request['others'],
                # [len(request['values']), self.feature_num, 1]
            }
        )
        print("******prediction:", pred.shape)
        return {
            'prediction': pred,
            'labels': request['labels'],
        }

    def postprocess(self, response: Dict) -> Dict:
        auc = roc_auc_score(response['labels'], response['prediction'])
        return {
            'auc': auc,
            'prediction': response['prediction'].tolist(),
        }
