# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage for Bayesian LoRA.
"""

import os
import sys
import peft
import hydra
import logging
import importlib
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any
from omegaconf import DictConfig
from torch.func import jacrev, functional_call
from torchmetrics import Accuracy, CalibrationError
from transformers.modeling_outputs import ModelOutput

from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    variance,
    stable_cholesky,
)
from utils import dsets
from utils.loggers import setup_loggers
from utils.setup_llm import setup_llm
from bayesian_lora.main import jacobian_mean


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="example_usage",
)
def main(cfg: DictConfig):
    #
    # 1. Load configuration from Hydra
    #
    device = "cuda:0"
    
    setup_loggers(cfg)
    os.makedirs(cfg.paths.output_dir, exist_ok=True) 

    #
    # 2. Load PEFT model and dataset
    #
    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    # model = model.to(device)
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dset = dset_class(tokenizer, add_space=cfg.llm.add_space)

    #
    # 3. Do MAP training
    #
    train_loader = dset.loader(
        is_s2s=cfg.llm.is_s2s,  # sequence to sequence model?
        batch_size=cfg.dset.train_bs,  # training batch size
        split=cfg.dset.train_split,  # training split name in dset
        subset_size=cfg.dset.train_subset,  # train on subset? (-1 = no subset)
    )
    map_param_path = f"{cfg.paths.output_dir}/MAP_params.pth"
    
    logging.info(f"Loading MAP parameters from {map_param_path}")
    del model
    llm_params = dict(cfg.llm) | {"use_peft": False}
    model, _, _ = setup_llm(**llm_params)
    model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
    model = model.to(device)


    #
    # 5. Calculate the (low-rank) Kronecker factors
    #
    def fwd_call(model: nn.Module, batch: Any) -> t.Tensor:
        prompts, _, _ = batch
        tok_kwargs = dict(cfg.tokenizer_run_kwargs) | {
            "padding": True,
            "return_tensors": "pt",
        }
        inputs = tokenizer(prompts, **tok_kwargs).to(device)
        outputs = model(**inputs)
        logits = (
            outputs.logits[:, dset.target_ids.squeeze(-1)]
            if cfg.llm.is_s2s
            else outputs.logits[:, -1, dset.target_ids.squeeze(-1)]
        )
        logits = logits.softmax(-1)
        return logits

    kfac_path = f"{cfg.paths.output_dir}/kronecker_factors.pth"
    logging.info("Computing the low-rank Kronecker factors")
    factors = calculate_kronecker_factors(
        model,
        fwd_call,
        train_loader,
        cfg.n_kfac,
        cfg.lr_threshold,
        ["lora"],
        use_tqdm=cfg.use_tqdm,
    )
    # Calculate Cholesky decomposition of the smaller factors
    factors = cholesky_decompose_small_factors(
        factors, cfg.lr_threshold, device, t.float32
    )
    t.save({"factors": factors}, kfac_path)

   


if __name__ == "__main__":
    main()
