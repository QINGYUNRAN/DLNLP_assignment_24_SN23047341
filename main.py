import os
import gc
from tqdm.auto import tqdm
import json
import numpy as np
import pandas as pd

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from datasets import Dataset, load_from_disk
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import re
from transformers import TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from scipy.special import softmax
from spacy.lang.en import English

# Toggle to use training set folds for model
debug_on_train_df = False   # 🔍 Setting debug_on_train_df to False to toggle using training set folds

# Enable to convert models for inference on-the-fly.
convert_before_inference = False   # ⚙️ Setting convert_before_inference to False to toggle model conversion for inference on-the-fly

# Temporary directory for saving datasets and intermediate files.
temp_data_folder = "/tmp/output/"