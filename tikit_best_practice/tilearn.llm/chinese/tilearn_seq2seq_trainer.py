# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


import numpy as np
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


import os, sys
from functools import partial
MAX_EVAL_SEQ_LENGTH = int(os.getenv('MAX_EVAL_SEQ_LENGTH', '128'))


#### TIACC 
#sys.path.append("../../../../../tilearn_llm/src/")
# from tilearn.llm.trainer import TrainerTiacc as Trainer
TIACC_TRAINING_DYNAMIC_ZERO = int(os.getenv('TIACC_TRAINING_DYNAMIC_ZERO', '0'))

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

# Metric
def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds

    #if isinstance(preds, tuple):
    #    preds = preds[0]

    #labels = labels[:, 1:]#.reshape(-1)
    #preds = preds[:, :-1]#.reshape(-1)


    #output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #output = map(lambda p, o: o.replace(p.replace('<s>', '').strip(), '').strip(), prompt, output)
    #output = list(output)
    #print(f"input_ids:{input_ids}, output_ids:{output_ids}, prompt:{prompt}, output:{output}")


    #print(f"preds:{preds}")

    ignore_pad_token_for_loss = True
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    print_num = 0

    for pred, label in zip(decoded_preds, decoded_labels):

        #pred = output.replace(prompt, '').strip()
        pred = pred.split("### Response:")[-1].strip()
        if len(pred) == 0:
            pred = "Null"
            print("pred is Null")

        if print_num < 2:
        #if 1:
            print(f"pred:{pred} label:{label}")
            print_num += 1

        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        result = scores[0]
        
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict
    #return {}

class TilearnSeq2SeqTrainer(Trainer):

    def __init__(self, *args, **kwargs):

        tokenizer = kwargs.pop('tokenizer', None)
        assert tokenizer is not None, "tokenizer is not set in trainer init!!!"
        compute_metrics_partial = partial(compute_metrics, tokenizer)


        compute_metrics_func = kwargs.pop('compute_metrics', None)
        if compute_metrics_func is None:
            kwargs["compute_metrics"] = compute_metrics_partial

        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["input_ids"]

        #if TIACC_TRAINING_DYNAMIC_ZERO == 1:
        #    generate_func = model.module.generate
        #else:
        #    generate_func = model.generate

        #print("run model.generate!!!")
        with torch.no_grad():
            #output_ids = generate_func(
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_EVAL_SEQ_LENGTH,
                temperature=1,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15
                ).cuda()

        #decoded_preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #print(f"------------- decoded_preds:{decoded_preds}")

        #print(f"output_ids:{output_ids}")
        #output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #prompt = self.tokenizer.batch_decode(input_ids)
        #output = map(lambda p, o: o.replace(p.replace('<s>', '').strip(), '').strip(), prompt, output)
        #output = list(output)
        #print(f"input_ids:{input_ids}, output_ids:{output_ids}, prompt:{prompt}, output:{output}")

        #output_ids = self.tokenizer(output, 
        #                            return_attention_mask=False, 
        #                            max_length=self.args.max_seq_length, 
        #                            truncation=True, 
        #                            padding=True,
        #                            return_tensors='pt').to('cuda')
        #output_ids = output_ids['input_ids']


        #decoded_preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #print(f"output:{output}, output_ids:{output_ids}, decoded_preds:{decoded_preds}")
        #exit()

        loss = torch.tensor([1,]).to('cuda')
        # XXX: adapt synced_gpus for fairscale as well
        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None

        return (loss, output_ids, labels)
