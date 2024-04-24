import os
from func.metrics import MetricsComputer
from transformers.trainer import Trainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification


def evaluate_on_test_set(output_dir, n_splits, tokenizer, test_ds, label2id, id2label, eval_encoder, EVAL_BATCH_SIZE):
    test_encoded = test_ds.map(eval_encoder, num_proc=os.cpu_count())
    all_folds_metrics = []

    for fold_idx in range(n_splits):
        model_path = os.path.join(output_dir, f"fold_{fold_idx}", "best")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        eval_args = TrainingArguments(
            report_to="none",
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            dataloader_drop_last=False
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=test_encoded,
            tokenizer=tokenizer,
            compute_metrics=MetricsComputer(eval_ds=test_encoded, label2id=label2id, id2label=id2label),
            data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
        )
        eval_res = trainer.evaluate()

        all_folds_metrics.append(eval_res)

        print(f"Fold {fold_idx} metrics on test set: {eval_res}")

    return all_folds_metrics