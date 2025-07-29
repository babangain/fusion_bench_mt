import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from .utils import state_dict_to_vector, vector_to_state_dict
from fusion_bench import BaseModelPool

log = logging.getLogger(__name__)


def trainable_state_dict(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: param for name, param in module.named_parameters() if param.requires_grad
    }


def state_dict_sub(dict_a: Dict[str, Tensor], dict_b: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: dict_a[k] - dict_b[k] for k in dict_a if k in dict_b}


class TaskArithmeticWithTrustRegionForCausalLM:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        scaling_factor: Union[float, List[float]],
        threshold_quantile: float,
        max_samples: int,
        batch_size: int,
        zero_shot: bool,
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.scaling_factor = scaling_factor
        self.threshold_quantile = threshold_quantile
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.zero_shot = zero_shot
        self.device = device

def run(
    self,
    modelpool: BaseModelPool,
    train_datasets: Union[None, Dict[str, List[str]]] = None
):
    # Load pretrained model and fine-tuned models
    pretrained_model = modelpool.load_pretrained_model()
    pretrained_model = deepcopy(pretrained_model).to(self.device).eval()
    pretrained_sd = trainable_state_dict(pretrained_model)

    finetuned_models = {
        name: model.to(self.device).eval()
        for name, model in modelpool.named_models()
    }

    # Compute task vectors
    task_vectors = {
        name: state_dict_sub(trainable_state_dict(model), pretrained_sd)
        for name, model in finetuned_models.items()
    }
    task_vectors = {name: state_dict_to_vector(vec) for name, vec in task_vectors.items()}

    # Compute trust region vectors
    if self.zero_shot:
        all_avg_abs_grads = {name: tv.abs() for name, tv in task_vectors.items()}
    else:
        if train_datasets is None:
            raise ValueError("train_datasets must be provided when zero_shot=False.")
        all_avg_abs_grads = self.compute_avg_abs_grads(pretrained_model, train_datasets)
        all_avg_abs_grads = {name: state_dict_to_vector(g) for name, g in all_avg_abs_grads.items()}

    # Trust region overlap
    Omega = torch.zeros_like(next(iter(all_avg_abs_grads.values())))
    for i in all_avg_abs_grads:
        for j in all_avg_abs_grads:
            if i != j:
                Omega += all_avg_abs_grads[i] * task_vectors[j].abs()

    values, _ = Omega.sort(descending=False)
    threshold_idx = int(Omega.numel() * self.threshold_quantile)
    threshold = values[min(threshold_idx, Omega.numel() - 1)]
    mask = Omega < threshold

    # Apply mask
    for task in task_vectors:
        task_vectors[task] = task_vectors[task] * mask

    task_vector_sum = sum(task_vectors.values())
    task_vector_sum = vector_to_state_dict(task_vector_sum, pretrained_sd)

    # Merge
    if isinstance(self.scaling_factor, (float, int)):
        for name, param in pretrained_model.named_parameters():
            if name in task_vector_sum:
                param.data += task_vector_sum[name].to(param.device) * self.scaling_factor
        return pretrained_model
    elif isinstance(self.scaling_factor, Iterable):
        models = {}
        for sf in self.scaling_factor:
            model = deepcopy(pretrained_model)
            for name, param in model.named_parameters():
                if name in task_vector_sum:
                    param.data += task_vector_sum[name].to(param.device) * sf
            models[sf] = model
        return models
    else:
        raise ValueError("Invalid type for scaling_factor.")

    def compute_avg_abs_grads(
        self, pretrained_model: nn.Module, train_datasets: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Tensor]]:
        pretrained_model.train()
        all_avg_abs_grads = {}

        for task_name, texts in train_datasets.items():
            dataloader = DataLoader(texts, batch_size=self.batch_size, shuffle=True)
            grad_accum = defaultdict(float)
            num_samples = 0

            for batch in dataloader:
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
                outputs = pretrained_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                pretrained_model.zero_grad()
                loss.backward()

                for name, param in pretrained_model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        grad_accum[name] += param.grad.abs().detach().cpu()

                num_samples += len(batch)
                if num_samples >= self.max_samples:
                    break

            avg_grads = {k: v / num_samples for k, v in grad_accum.items()}
            all_avg_abs_grads[task_name] = avg_grads

        return all_avg_abs_grads
