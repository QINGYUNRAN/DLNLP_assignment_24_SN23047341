import os
import json
from itertools import chain
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', "Dataset")
train_data_path = os.path.join(data_path, 'train.json')
test_data_path = os.path.join(data_path, "test.json")

def get_data(train_data_path, test_data_path):
    data = json.load(open(train_data_path))
    # test_data = json.load(open(test_data_path))
    all_labels = [
        'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME',
        'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
    ]
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    target = [l for l in all_labels if l != "O"]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in train_data],
        "document": [str(x["document"]) for x in train_data],
        "tokens": [x["tokens"] for x in train_data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in train_data],
        "provided_labels": [x["labels"] for x in train_data],
    })
    test_ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in test_data],
        "document": [str(x["document"]) for x in test_data],
        "tokens": [x["tokens"] for x in test_data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in test_data],
        "provided_labels": [x["labels"] for x in test_data],
    })

    return train_ds, test_ds, label2id, id2label, target