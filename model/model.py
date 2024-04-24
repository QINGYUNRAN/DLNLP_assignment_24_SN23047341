from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from peft.mapping import get_peft_config, get_peft_model


class ModelInit:
    def __init__(
        self,
        checkpoint,
        id2label,
        label2id,
        peft_config,
    ):
        self.checkpoint = checkpoint
        self.id2label = id2label
        self.label2id = label2id
        self.peft_config = peft_config

    def __call__(self):
        base_model = DebertaV2ForTokenClassification.from_pretrained(
            self.checkpoint,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        model = get_peft_model(base_model, self.peft_config)
        model.print_trainable_parameters()
        return model
