"""
multitask.py
------------
A single shared encoder with two classification heads (department and
urgency).

This is the architecture compared against the two-separate-models
baseline in Tier 1 experiment 3. The research argument is that
department and urgency are correlated views of the same inquiry -- an
"I cannot log in before my exam tomorrow" message is simultaneously an
IT issue and a high-urgency one -- so a shared representation may
transfer between the tasks while halving the parameter count and the
number of forward passes at inference.

save() and load() mirror each other so that whatever wrote a checkpoint
to models/multitask/ -- this project no longer ships a training script,
but the format is what one would need to produce -- and the deployed
service agree on exactly the same class definition; a mismatch between
the two is the usual cause of silently wrong weights being loaded.

Moved from app/multitask_model.py, with one change: load() now passes
weights_only=True to torch.load. The old call unpickled arbitrary objects
from pytorch_model.bin, which is remote code execution if a weights
directory ever arrives from outside the project.
"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MultiTaskClassifier(nn.Module):
    """Shared transformer encoder + one linear head per task."""

    def __init__(self, base_model: str, num_category_labels: int, num_urgency_labels: int,
                 dropout: float = 0.2):
        super().__init__()
        self.base_model_name = base_model
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden, num_category_labels)
        self.urgency_head = nn.Linear(hidden, num_urgency_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                category_labels=None, urgency_labels=None, task_weights=(1.0, 1.0)):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        category_logits = self.category_head(pooled)
        urgency_logits = self.urgency_head(pooled)

        loss = None
        if category_labels is not None and urgency_labels is not None:
            fn = nn.CrossEntropyLoss()
            # Joint objective: the sum of both task losses. task_weights
            # lets you trade the tasks off if one dominates.
            loss = (task_weights[0] * fn(category_logits, category_labels)
                    + task_weights[1] * fn(urgency_logits, urgency_labels))

        return {"loss": loss, "category_logits": category_logits,
                "urgency_logits": urgency_logits, "pooled": pooled}

    def get_input_embeddings(self):
        """Exposed so Integrated Gradients can attribute to token embeddings."""
        return self.encoder.get_input_embeddings()

    # -- persistence -------------------------------------------------------

    def save(self, directory, tokenizer, category_labels, urgency_labels):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "pytorch_model.bin")
        tokenizer.save_pretrained(directory)
        with open(directory / "multitask_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": self.base_model_name,
                    "category_labels": list(category_labels),
                    "urgency_labels": list(urgency_labels),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory):
        """Rebuilds the model from a saved directory.

        Returns (model, tokenizer, category_labels, urgency_labels).
        """
        directory = Path(directory)
        with open(directory / "multitask_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Build the skeleton from the *config* rather than downloading
        # pretrained weights -- the state dict overwrites them anyway.
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.base_model_name = cfg["base_model"]
        encoder_config = AutoConfig.from_pretrained(cfg["base_model"])
        model.encoder = AutoModel.from_config(encoder_config)
        hidden = encoder_config.hidden_size
        model.dropout = nn.Dropout(0.2)
        model.category_head = nn.Linear(hidden, len(cfg["category_labels"]))
        model.urgency_head = nn.Linear(hidden, len(cfg["urgency_labels"]))

        # weights_only=True: load tensors, refuse to execute pickled code.
        state = torch.load(
            directory / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(directory)
        return model, tokenizer, cfg["category_labels"], cfg["urgency_labels"]


def mean_pool(last_hidden_state, attention_mask):
    """Mask-aware mean pooling over token representations.

    Preferred over taking [CLS] because the encoder here is not
    necessarily trained with a next-sentence objective, and mean pooling
    gives a more stable sentence vector for the duplicate-detection
    feature that reuses these embeddings.
    """
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
