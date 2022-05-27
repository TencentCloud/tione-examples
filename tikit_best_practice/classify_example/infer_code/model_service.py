import os
from typing import Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import Net

import numpy as np
import cv2
import tiinfer
import base64
import urllib
import torchvision.models as models

import requests as req
from PIL import Image
from io import BytesIO
from torchvision import transforms, utils
import hashlib
import json

PYTORCH_FILE = "model/model_best.pth.tar"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

img_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

class MnistClassifyModel(tiinfer.Model):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.model_dir = model_dir
        self.model_name = "classification-demo"
        self.result_type = "classification"
        self.ready = False
        self.model = None
        self.imgsz= 224 #28
        self.mean = [0.485, 0.456, 0.406] #0.1307
        self.std = [0.229, 0.224, 0.225] #0.3081
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        with open("./mapping.json",'r') as load_f:
            self.target_mapping = json.load(load_f)

    def load(self) -> bool:
        try:
            # Load  model file
            num_classes=102
            model = models.__dict__["resnet50"]()
            num_fc_feat = model.fc.in_features
            model.fc = nn.Linear(num_fc_feat, num_classes)
            model = torch.nn.DataParallel(model)
            model.to(self.device)

            model_file = os.path.join(self.model_dir,  PYTORCH_FILE)
            checkpoint = torch.load(model_file, map_location=self.device)

            model.load_state_dict(checkpoint['state_dict'])

            model.eval()
            self.model = model
            logging.info("load model to device %s" % (self.device) )
        except Exception as e:  # pylint: disable=broad-except
            logging.error("load model failed: %s", e)
            self.ready = False
            return self.ready
        self.ready = True
        return self.ready

    def preprocess(self, request: Dict) -> Dict:
        try:
            if "image" not in request:
                return {"error": "invalid param"}
            img_data = request["image"]
            
            if img_data.startswith("http"):
                logging.info("image url %s", img_data)
                response = req.get(img_data)
                instances = response.content
            else:
                instances = base64.b64decode(img_data)

            return {"instances": [instances]}
        except Exception as e:
            logging.error("Preprocess failed: %s" % (e))
            return {"error": "preprocess failed"}

    def predict(self, request: Dict) -> Dict:
        if "instances" not in request :
            return request
        with torch.no_grad():
            try:
                img_pil = Image.open(BytesIO(request["instances"][0]))
                img_pil = img_pil.resize((224,224))
                img_tensor = img_preprocess(img_pil)
                
                output = self.model(img_tensor.unsqueeze(0))
                pred = output.argmax(dim=1, keepdim=False)            
                return {
                    "pred_out":  pred.to('cpu').tolist()
                }
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed to predict" % (e))
                request.pop("image", '')
                request.pop("instances", '')
                request["predict_err"] = str(e)
                return request

    def postprocess(self, request: Dict) -> Dict:
        if "pred_out" not in request:
            return request
        try:
            pred_out = request["pred_out"]
            
            return {"result": {
                "type": self.result_type,
                "pred": self.target_mapping[str(pred_out[0])]
            }}
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Postprocess failed: %s" % (e))
            request.pop("image", '')
            request.pop("instances", '')
            request.pop("predictions", '')
            request["post_process_err"] = str(e)
            return request
