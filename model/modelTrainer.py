import json
import copy
import gc
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd
from spacy.lang.en import English
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorForTokenClassification
from peft.mapping import get_peft_config, get_peft_model
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from peft.tuners.lora import LoraConfig
from peft.utils import TaskType
from datasets import Dataset, DatasetDict, concatenate_datasets

TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 2048
EVAL_MAX_LENGTH = 3072
CONF_THRESH = 0.9
LR = 5e-4  # Note: lr for LoRA should be order of magnitude larger than usual fine-tuning
LR_SCHEDULER_TYPE = "linear"
NUM_EPOCHS = 3
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 16 // BATCH_SIZE
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FREEZE_EMBEDDING = False
FREEZE_LAYERS = 6
LORA_R = 16  # rank of the A and B matricies, the lower the more efficient but more approximate
LORA_ALPHA = LORA_R * 2  # alpha/r is multiplied to BA
AMP = True
N_SPLITS = 4
NEGATIVE_RATIO = 0.3  # downsample ratio of negative samples in the training set
OUTPUT_DIR = "output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=AMP,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=50,
    eval_delay=100,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    metric_for_best_model="f5",
    greater_is_better=True,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
)

target_modules = ["key_proj", "value_proj", "query_proj", "dense", "classifier"]  # all linear layers
if not FREEZE_EMBEDDING:
    target_modules.append("word_embeddings")  # embedding layer
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    target_modules=target_modules,
    modules_to_save=["classifier"],  # The change matrices depends on this randomly initialized matrix, so this needs to be saved
    layers_to_transform=[i for i in range(FREEZE_LAYERS, 24)],
    r=LORA_R,  # The dimension of the low-rank matrices. The higher, the more accurate but less efficient
    lora_alpha=LORA_ALPHA,  # The scaling factor for the low-rank matrices.
    lora_dropout=0.1,
    bias="lora_only"
)