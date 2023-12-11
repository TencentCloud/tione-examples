import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union, List
import datasets
import torch
import logging
from datasets import load_dataset, concatenate_datasets
import copy
import transformers
import random

IGNORE_INDEX = -100

import os
#DATA_INPUT_COLUMN = os.getenv('DATA_INPUT_COLUMN', 'input')
#DATA_OUTPUT_COLUMN = os.getenv('DATA_OUTPUT_COLUMN', 'output')
DATA_INSTRUCTION = os.getenv('DATA_INSTRUCTION', "")


logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

def buid_dataset_train(data_path: str,
                       tokenizer: transformers.PreTrainedTokenizer,
                       max_seq_length: int, data_cache_dir = None,
                       preprocessing_num_workers = None,
                       ):

    logging.warning("building dataset train...")
    all_datasets = []

    extension = data_path.split(".")[-1]
    if extension == "txt":
            extension = "text"

    raw_dataset = load_dataset(
        extension,
        data_files=data_path,
        cache_dir=data_cache_dir,
        )


    column_names_dict = raw_dataset.column_names
    file_name = list(column_names_dict.keys())[0]
    column_names = column_names_dict[file_name]
    print(f"TIACC DATA - column_names: {column_names}")
    input_column_name = column_names[0]
    output_column_name = column_names[1]

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        #for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
        for input, output in zip(examples[input_column_name], examples[output_column_name]):
            if input is not None and input !="":
                instruction = DATA_INSTRUCTION 
                #instruction = instruction+'\n'+input
                instruction = instruction + input
                #instruction = input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            #input_ids = torch.LongTensor(s + t)[:max_seq_length]
            #labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            #assert len(input_ids) == len(labels)

            input_ids_t = (s + t)[:max_seq_length]
            labels_t = ([IGNORE_INDEX] * len(s) + t)[:max_seq_length]

            pad_len = max_seq_length - len(input_ids_t)
            input_ids_t = input_ids_t + [tokenizer.pad_token_id] * pad_len

            pad_len = max_seq_length - len(labels_t)
            # labels_t = labels_t + [tokenizer.pad_token_id] * pad_len
            labels_t = labels_t + [IGNORE_INDEX] * pad_len

            input_ids = torch.LongTensor(input_ids_t)
            labels = torch.LongTensor(labels_t)

            assert len(input_ids) == len(labels)

            all_input_ids.append(input_ids)
            all_labels.append(labels)


        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results

    #################################################################
    tokenization_func = tokenization
    tokenized_dataset = raw_dataset.map(
        tokenization_func,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        keep_in_memory=False,
        desc="preprocessing on dataset",
        )

    tokenized_dataset.set_format('torch')
    processed_dataset = tokenized_dataset[file_name]

    print(f"train - input_ids: {processed_dataset[0]['input_ids']}")
    print(f"train - labels: {processed_dataset[0]['labels']}")

    return processed_dataset

def buid_dataset_eval(data_path: str,
                      tokenizer: transformers.PreTrainedTokenizer,
                      max_seq_length: int, data_cache_dir = None,
                      preprocessing_num_workers = None,
                      ):

    logging.warning("building dataset eval...")
    all_datasets = []

    extension = data_path.split(".")[-1]
    if extension == "txt":
            extension = "text"

    raw_dataset = load_dataset(
        extension,
        data_files=data_path,
        cache_dir=data_cache_dir,
        )

    column_names_dict = raw_dataset.column_names
    file_name = list(column_names_dict.keys())[0]
    column_names = column_names_dict[file_name]
    print(f"TIACC DATA - column_names: {column_names}")
    input_column_name = column_names[0]
    output_column_name = column_names[1]

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        #for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
        for input, output in zip(examples[input_column_name], examples[output_column_name]):
            if input is not None and input !="":
                instruction = DATA_INSTRUCTION 
                #instruction = instruction+'\n'+input
                instruction = instruction + input
                #instruction = input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            #input_ids = torch.LongTensor(s + t)[:max_seq_length]
            #labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            #assert len(input_ids) == len(labels)

            input_ids_t = (s)[:max_seq_length]
            labels_t = (t)[:max_seq_length]

            input_ids = torch.LongTensor(input_ids_t)
            labels = torch.LongTensor(labels_t)

            all_input_ids.append(input_ids)
            all_labels.append(labels)


        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results

    #################################################################
    tokenization_func = tokenization
    tokenized_dataset = raw_dataset.map(
        tokenization_func,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        keep_in_memory=False,
        desc="preprocessing on dataset",
        )
    
    tokenized_dataset.set_format('torch')
    processed_dataset = tokenized_dataset[file_name]

    print(f"eval - input_ids: {processed_dataset[0]['input_ids']}")
    print(f"eval - labels: {processed_dataset[0]['labels']}")

    return processed_dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
