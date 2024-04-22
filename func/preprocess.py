import pandas as pd
import os



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
