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
from omegaconf import DictConfig, OmegaConf
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

    # Brief info
    # TODO try a different model than GPT2 -> getting weird results 
    # TODO export selected config as a dict - specially: dataset name, lora params, 
    # batch size, learning rate, training data size, train / valid / test split? 

    print('Dataset:', cfg.dset.name)
    print('Model: ', cfg.llm.name)
    print('Lora parameters: ', cfg.llm.peft.r)
    # print(OmegaConf.to_yaml(cfg))  # Print the whole config

    # Load PEFT model and dataset
    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dset = dset_class(tokenizer, add_space=cfg.llm.add_space)

    # Do MAP training
    train_loader = dset.loader(
        is_s2s=cfg.llm.is_s2s,  # sequence to sequence model?
        batch_size=cfg.dset.train_bs,  # training batch size
        split=cfg.dset.train_split,  # training split name in dset
        subset_size=cfg.dset.train_subset,  # train on subset? (-1 = no subset)
    )
    map_param_path = f"{cfg.paths.output_dir}/MAP_params.pth"
    grad_steps, epoch = 0, 0

    # setup optimiser
    opt_cfg = dict(cfg.opt)
    # add prior / regularization for MAP objective:
    opt_cfg |= {"weight_decay": 1 / cfg.prior_var}
    optclass = getattr(
        importlib.import_module(opt_cfg.pop("module")),
        opt_cfg.pop("classname"),
    )
    opt = optclass(model.parameters(), **opt_cfg)
    logging.info("Training MAP parameters")

    # TODO export sample prompts and classes dsdsdsd
    # TODO export loss, train step, batch size
    losses = []
    samples = []

    # train_steps = cfg.train_steps
    train_steps = 10

    while grad_steps < train_steps:
        epoch += 1
        logging.info(f"Beginning epoch {epoch} ({grad_steps} / {train_steps})")
        for batch in tqdm(train_loader, disable=not cfg.use_tqdm, file=sys.stdout):
            opt.zero_grad()
            prompts, classes, _ = batch
            
            inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)
            logits = model(**inputs).logits[:, -1, dset.target_ids.squeeze(-1)]
            # loss = F.cross_entropy(logits[:, -1], targets.to(device))
            loss = F.cross_entropy(logits, classes.to(device))
            assert not t.isnan(loss).any(), "NaN in loss for MAP training."
            loss.backward()
            opt.step()
            grad_steps += 1
            
            losses.append(loss.cpu().detach().numpy().tolist())

            if grad_steps < 10:
                samples.append([prompts, classes.numpy().tolist()])
            
            if not grad_steps < train_steps:
                break
    logging.info(f"Saving MAP parameters after finetuning to {map_param_path}")
    model.save_pretrained(map_param_path)

    print('losses', losses[0])
    print('samples', samples[0])


if __name__ == "__main__":
    main()
