from func.get_data import get_data
from func.global_seed import global_seed
from func.tokenize import CustomTokenizer
import os
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.training_args import TrainingArguments
from peft.tuners.lora import LoraConfig
from peft.utils import TaskType
from datasets import Dataset, DatasetDict, concatenate_datasets
from model.model import ModelInit
from run.train import train
from run.test import evaluate_on_test_set
import argparse




def setup_args():
    parser = argparse.ArgumentParser(description="Setup for model training and evaluation.")

    # Model and training parameters
    parser.add_argument("--training_max_length", default=2048, type=int,
                        help="Maximum length of the training sequences.")
    parser.add_argument("--eval_max_length", default=3072, type=int, help="Maximum length of the evaluation sequences.")
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str, help="Type of learning rate scheduler.")
    parser.add_argument("--num_epochs", default=3, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=1, type=int, help="Training batch size.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Evaluation batch size.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Warmup ratio for the scheduler.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--freeze_layers", default=6, type=int, help="Number of transformer layers to freeze.")
    parser.add_argument("--lora_r", default=16, type=int, help="Rank for LoRA adaptation.")
    parser.add_argument("--amp", default=True, type=bool, help="Automatic mixed precision training.")
    parser.add_argument("--n_splits", default=4, type=int, help="Number of splits for cross-validation.")
    parser.add_argument("--negative_ratio", default=0.3, type=float, help="Ratio of negatives for training.")
    parser.add_argument("--output_dir", default="output", type=str, help="Directory for saving outputs.")
    parser.add_argument("--data_folder", default="Dataset", type=str,
                        help="Folder where training and test data are stored.")

    return parser.parse_args()


def main():
    args = setup_args()

    global_seed(42)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, args.data_folder)
    train_data_path = os.path.join(data_path, 'train.json')
    test_data_path = os.path.join(data_path, 'test.json')
    log_path = os.path.join(dir_path, "logs")
    training_model_path = "microsoft/deberta-v3-large"
    target_modules = ["key_proj", "value_proj", "query_proj", "dense", "classifier", "word_embeddings"]
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        target_modules=target_modules,
        modules_to_save=["classifier"],
        layers_to_transform=[i for i in range(args.freeze_layers, 24)],
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        bias="lora_only"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=args.amp,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=16 // args.batch_size,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=50,
        eval_delay=100,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_dir=log_path,
        logging_steps=10,
        metric_for_best_model="f5",
        greater_is_better=True,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    tokenizer = DebertaV2TokenizerFast.from_pretrained(training_model_path)
    train_ds, test_ds, label2id, id2label, target = get_data(train_data_path, test_data_path)
    train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=args.training_max_length)
    eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=args.eval_max_length)

    model = ModelInit(
        training_model_path,
        id2label=id2label,
        label2id=label2id,
        peft_config=lora_config
    )
    all_fold_train_losses, all_fold_eval_precisions = train(training_args, model, train_ds, tokenizer, train_encoder,
                                                            eval_encoder, label2id, id2label, args.negative_ratio,
                                                            args.n_splits)
    all_metrics = evaluate_on_test_set(args.output_dir, args.n_splits, tokenizer, test_ds, label2id, id2label,
                                       eval_encoder, args.eval_batch_size)


if __name__ == '__main__':
    main()