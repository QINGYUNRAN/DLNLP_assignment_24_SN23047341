import os
import json
from itertools import chain
import pandas as pd

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, "Dataset")
train_data_path = os.path.join(data_path, 'train.json')
test_data_path = os.path.join(data_path, "test.json")

def get_data(train_data_path, test_data_path, fold):
    data = json.load(open(train_data_path))
    test_data = json.load(open(test_data_path))
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    # Create a DataFrame for the training data using the loaded training data.
    df_train = pd.DataFrame(data)

    # Add a new column 'fold' to the training DataFrame, representing the fold number.
    df_train['fold'] = df_train['document'] % fold

    # Create a DataFrame for the test data using the loaded test data.
    df_test = pd.DataFrame(test_data)

def find_span(target, document):
    idx = 0
    spans = []
    span = []

    # Iterate through the document tokens
    for i, token in enumerate(document):
        # If the current token doesn't match the target start anew
        if token != target[idx]:
            idx = 0
            span = []
            continue
        # If token matches, append its index to the span list
        span.append(i)
        idx += 1
        # If the entire target is found, append the span to the list of spans
        if idx == len(target):
            spans.append(span)
            # Reset span and idx for next potential match
            span = []
            idx = 0
            continue
    return spans


def downsample_df(train_df, percent):
    # Add a new column 'is_labels' to indicate if labels are present in the sample
    train_df['is_labels'] = train_df['labels'].apply(lambda labels: any(label != 'O' for label in labels))

    # Separate samples with labels and samples without labels
    true_samples = train_df[train_df['is_labels'] == True]
    false_samples = train_df[train_df['is_labels'] == False]

    # Calculate the number of false samples to keep after downsampling
    n_false_samples = int(len(false_samples) * percent)

    # Randomly sample false samples to downsample
    downsampled_false_samples = false_samples.sample(n=n_false_samples, random_state=42)

    # Concatenate true samples and downsampled false samples to create the downsampled DataFrame
    downsampled_df = pd.concat([true_samples, downsampled_false_samples])

    return downsampled_df