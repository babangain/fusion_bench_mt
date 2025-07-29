
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from .utils import state_dict_to_vector, vector_to_state_dict, state_dict_sub

log = logging.getLogger(__name__)

def trainable_state_dict(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: param for name, param in module.named_parameters() if param.requires_grad
    }

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

    def run(self, pretrained_model, finetuned_models: Dict[str, nn.Module], datasets: Dict[str, List[str]]):
        pretrained_model = deepcopy(pretrained_model).to(self.device).eval()
        pretrained_sd = trainable_state_dict(pretrained_model)
        task_vectors = {
            name: state_dict_sub(trainable_state_dict(model), pretrained_sd)
            for name, model in finetuned_models.items()
        }
        task_vectors = {k: state_dict_to_vector(v) for k, v in task_vectors.items()}

        if not self.zero_shot:
            all_avg_abs_grads = self.compute_avg_abs_grads(pretrained_model, datasets)
            all_avg_abs_grads = {k: state_dict_to_vector(v) for k, v in all_avg_abs_grads.items()}
        else:
            all_avg_abs_grads = {name: tv.abs() for name, tv in task_vectors.items()}

        # Trust region
        Omega = torch.zeros_like(next(iter(all_avg_abs_grads.values())))
        for i in all_avg_abs_grads:
            for j in all_avg_abs_grads:
                if i != j:
                    Omega += all_avg_abs_grads[i] * task_vectors[j].abs()

        values, _ = Omega.sort(descending=False)
        threshold_idx = int(Omega.numel() * self.threshold_quantile)
        threshold = values[min(threshold_idx, Omega.numel() - 1)]
        mask = Omega < threshold

        for task in task_vectors:
            task_vectors[task] = task_vectors[task] * mask

        task_vector_sum = sum(task_vectors.values())
        task_vector_sum = vector_to_state_dict(task_vector_sum, pretrained_sd)

        if isinstance(self.scaling_factor, (int, float)):
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
            raise ValueError("Invalid scaling_factor type")

    def compute_avg_abs_grads(self, model, datasets: Dict[str, List[str]]):
        model.train()
        grads = {}

        for task_name, texts in datasets.items():
            dataloader = DataLoader(texts, batch_size=self.batch_size, shuffle=True)
            grad_accum = defaultdict(lambda: torch.zeros_like(next(model.parameters())))

            num_samples = 0
            for batch in dataloader:
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                model.zero_grad()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        grad_accum[name] += param.grad.abs().detach().cpu()
                num_samples += len(batch)
                if num_samples >= self.max_samples:
                    break

            grads[task_name] = {k: v / num_samples for k, v in grad_accum.items()}

        return grads
