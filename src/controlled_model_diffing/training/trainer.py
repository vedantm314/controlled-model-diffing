"""The one HF Trainer customisation this project needs: a fixed document
order during training."""
from __future__ import annotations

from torch.utils.data import SequentialSampler
from transformers import Trainer


class SequentialSamplerTrainer(Trainer):
    """Replaces Trainer's default (re)shuffling sampler with a sequential one
    over the pre-shuffled dataset. Without this, the Trainer's own seed would
    reshuffle document order at train time and recontaminate seed_data with
    seed_model — the two are supposed to vary independently (see
    recipe.py / the --seed-data / --seed-model CLI split)."""

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(ds)
