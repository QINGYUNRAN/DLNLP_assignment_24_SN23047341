import numpy as np
import os
import json
import torch
import gc
from pathlib import Path
from transformers.trainer import Trainer
from func.metrics import MetricsComputer, MetricsCallback
from transformers.data.data_collator import DataCollatorForTokenClassification
dir_path = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(dir_path, "checkpoint")
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def train(args, model, train_ds, tokenizer, train_encoder, eval_encoder, label2id, id2label, NEGATIVE_RATIO, N_SPLITS):
    metrics_callback = MetricsCallback()
    folds = [
        (
            np.array([i for i, d in enumerate(train_ds["document"]) if int(d) % N_SPLITS != s]),
            np.array([i for i, d in enumerate(train_ds["document"]) if int(d) % N_SPLITS == s])
        )
        for s in range(N_SPLITS)
    ]
    all_eval_metrics = []
    # best_auc = 0
    # best_model = None
    # best_fold_idx = -1
    all_fold_train_losses = []
    all_fold_eval_precisions = []
    negative_idxs = [i for i, labels in enumerate(train_ds["provided_labels"]) if not any(np.array(labels) != "O")]
    exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]
    for fold_idx, (train_idx, eval_idx) in enumerate(folds):
        args.run_name = f"fold-{fold_idx}"
        args.output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
        tds = train_ds.select([i for i in train_idx if i not in exclude_indices])
        tds = tds.map(train_encoder, num_proc=os.cpu_count())
        eval_ds = train_ds.select(eval_idx)
        eval_ds = eval_ds.map(eval_encoder, num_proc=os.cpu_count())
        trainer = Trainer(
            args=args,
            model_init=model,
            train_dataset=tds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=MetricsComputer(eval_ds=eval_ds, label2id=label2id, id2label=id2label),
            data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
            callbacks=[metrics_callback],
        )
        trainer.train()
        eval_res = trainer.evaluate(eval_dataset=eval_ds)
        all_fold_train_losses.append(metrics_callback.train_losses)
        all_fold_eval_precisions.append(metrics_callback.eval_metrics['eval_precision'])
        all_eval_metrics.append(eval_res)

        # if 'eval_roc_auc' in eval_res:
        #     fold_auc = eval_res['eval_roc_auc']
        #     if fold_auc > best_auc:
        #         best_auc = fold_auc
        #         best_model = trainer.model
        #         best_fold_idx = fold_idx
        # metrics_callback.reset()

        with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
            json.dump(eval_res, f)
        trainer.model = trainer.model.base_model.merge_and_unload()
        trainer.save_model(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "best"))
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    for idx, metrics in enumerate(all_eval_metrics):
        print(
            f"Fold {idx}: Precision={metrics['eval_precision']}, Recall={metrics['eval_recall']}, F1 Score={metrics['eval_f1']}")
    # if best_model:
    #     print(f"Best Fold: {best_fold_idx}, AUC: {best_auc}")
    return all_fold_train_losses, all_fold_eval_precisions
