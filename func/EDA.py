import json
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams["figure.figsize"] = (12, 8)


dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..',"Dataset")
train_data_path = os.path.join(data_path, 'train.json')
image_dir = os.path.join(dir_path, '..', "image_output")

def token_class(dataset):
    c = Counter([label for essay in dataset for label in essay["labels"]])
    c_positive = {k: v for k, v in c.items() if k != "O"}
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.barplot(
        x=list(c_positive.keys()),
        y=list(c_positive.values()),
        hue=list(c_positive.keys()),
        dodge=False,
        ax=ax,
    )
    ax.tick_params(axis="x", labelsize=7.8)
    ax.set_title("Count of PII positive classes in dataset")
    ax.get_legend()
    ax.grid()
    c_without_name = {
        k: v
        for k, v in c_positive.items()
        if k != "B-NAME_STUDENT" and k != "I-NAME_STUDENT"
    }
    fig.savefig(os.path.join(image_dir, "class.pdf"))
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.barplot(
        x=list(c_without_name.keys()),
        y=list(c_without_name.values()),
        hue=list(c_without_name.keys()),
        dodge=False,
        ax=ax,
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Count of PII positive classes in dataset without names included")
    ax.get_legend()
    ax.grid()
    fig.savefig(os.path.join(image_dir, "class_without_name.pdf"))

def plot_distribution(data_df):
    def plot_bar_chart(data, target_column, title, xlabel, ylabel):
        target = data[target_column].value_counts(sort=False).reset_index(name='total')
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.barplot(
            data=target,
            y='total',
            x=target_column,
            hue=target_column,
            estimator=lambda x: sum(x) * 100.00 / target['total'].sum())
        fig.tight_layout(pad=3.0)
        ax.set_xlabel(xlabel, fontdict={'weight': 'bold'})
        ax.set_ylabel(ylabel, fontdict={'weight': 'bold'})
        ax.set_title(title)
        # ax..xticks(rotation=0)
        # plt.tick_params(axis='both', which='major', labelsize=9)
        # show percentage on bar for first 3 bars
        for index, row in target.iterrows():
            y = row.total * 100.00 / target['total'].sum()
            ax.text(row.name, y + 0.15, f'{y:.2f}%', fontsize=9)
        ax.get_legend()
        fig.savefig(os.path.join(image_dir, "distribution.pdf"))

    def plot_non_pii_entity_doc(df):
        df['non_pii_entity_only'] = df['labels'].apply(lambda x: sum(label == 'O' for label in x) == len(x))
        plot_bar_chart(df, 'non_pii_entity_only', "% of Documents with Non-PII Entity ('O' Label) Only",
                       "Non-PII Entity", "Percentage")
        print(df['non_pii_entity_only'].value_counts())
    plot_non_pii_entity_doc(data_df)

def EDA():
    with open(train_data_path) as f:
        dataset = json.load(f)
    df_train = pd.read_json(train_data_path)
    print(f"Number of training points: {len(dataset)}")
    token_class(dataset)
    plot_distribution(df_train)
    return


if __name__ == '__main__':
    EDA()
