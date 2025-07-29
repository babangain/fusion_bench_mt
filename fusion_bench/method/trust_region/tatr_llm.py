import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from fusion_bench import BaseModelPool

log = logging.getLogger(__name__)


def trainable_state_dict(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: param.detach().clone() for name, param in module.named_parameters()
        if param.requires_grad
    }


def state_dict_sub(dict_a: Dict[str, Tensor], dict_b: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        k: dict_a[k].cpu().to(torch.bfloat16) - dict_b[k].cpu().to(torch.bfloat16)
        for k in dict_a if k in dict_b
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

    @torch.no_grad()
    def run(
        self,
        modelpool: BaseModelPool,
        train_datasets: Union[None, Dict[str, List[str]]] = None,
    ):
        # Load and prepare models
        pretrained_model = deepcopy(modelpool.load_pretrained_model()).to(self.device, dtype=torch.bfloat16).eval()
        pretrained_sd = trainable_state_dict(pretrained_model)

        finetuned_models = {
            name: model.to(self.device, dtype=torch.bfloat16).eval()
            for name, model in modelpool.named_models()
        }

        # Task vectors: difference of finetuned - pretrained
        task_vectors = {
            name: state_dict_sub(trainable_state_dict(model), pretrained_sd)
            for name, model in finetuned_models.items()
        }

        # Trust region gradients
        if self.zero_shot:
            all_avg_abs_grads = {
                name: {k: v.abs() for k, v in vec.items()} for name, vec in task_vectors.items()
            }
        else:
            if train_datasets is None:
                raise ValueError("train_datasets must be provided when zero_shot=False.")
            all_avg_abs_grads = self.compute_avg_abs_grads(pretrained_model, train_datasets)

        # Build Omega (trust region mask) in parameter-wise dict form
        param_keys = list(next(iter(task_vectors.values())).keys())
        Omega = {k: torch.zeros_like(v) for k, v in task_vectors[next(iter(task_vectors))].items()}

        for i in all_avg_abs_grads:
            for j in task_vectors:
                if i != j:
                    for name in Omega:
                        Omega[name] += all_avg_abs_grads[i][name] * task_vectors[j][name].abs()

        # Flatten all Omega tensors to compute global threshold
        all_values = torch.cat([v.flatten() for v in Omega.values()]).float()
        threshold = torch.quantile(all_values, self.threshold_quantile)


        # Build mask
        mask = {k: (v < threshold).to(torch.bfloat16) for k, v in Omega.items()}

        # Apply mask to each task vector
        for task in task_vectors:
            for name in task_vectors[task]:
                task_vectors[task][name] = task_vectors[task][name] * mask[name]

        # Sum masked vectors
        task_vector_sum = {k: torch.zeros_like(v) for k, v in pretrained_sd.items()}
        for vec in task_vectors.values():
            for name in vec:
                task_vector_sum[name] += vec[name]

        # Merge into models
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
        self,
        pretrained_model: nn.Module,
        train_datasets: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Tensor]]:
        all_avg_abs_grads = {}

        for task_name, texts in train_datasets.items():
            model = deepcopy(pretrained_model).to(self.device).train()
            grad_accum = defaultdict(lambda: 0)
            num_samples = 0

            dataloader = DataLoader(texts, batch_size=self.batch_size, shuffle=True)

            for batch in dataloader:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                model.zero_grad()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        grad = param.grad.detach().abs().cpu().to(torch.bfloat16)
                        if isinstance(grad_accum[name], Tensor):
                            grad_accum[name] += grad
                        else:
                            grad_accum[name] = grad

                num_samples += len(batch)
                if num_samples >= self.max_samples:
                    break

            avg_grads = {k: v / num_samples for k, v in grad_accum.items()}
            all_avg_abs_grads[task_name] = avg_grads

        return all_avg_abs_grads
