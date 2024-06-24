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
    
    cfg.print_config = False
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
    # 4. Evaluate the log likelihood
    #
    ll_path = f"{cfg.paths.output_dir}/ll.pth"
    logging.info(f"Loading LL from {ll_path}")
    LL = t.load(ll_path)

    #
    # 5. Calculate the (low-rank) Kronecker factors
    #

    kfac_path = f"{cfg.paths.output_dir}/kronecker_factors.pth"
    
    logging.info(f"Loading low-rank Kronecker factors from {kfac_path}")
    kfactors = t.load(kfac_path)
    factors = kfactors["factors"]

    #
    # 6. Use the marginal likelihood to optimise the prior variance
    #
    prior_path = f"{cfg.paths.output_dir}/prior_params.pth"
    logging.info("Loading prior parameters (optimised using marginal likelihood)")
    priors = t.load(prior_path)
    # s2 = t.Tensor([0.1]).to(device) #  priors["s2"]
    # s2 = priors["s2"]
    # TODO what is this number exactly? 
    s2 = 3e-4  # TODO try different values
    s2 = priors["s2"] / priors["s2"] * s2
    print('Ignoring optimized prior - using the default value')

    #
    # 7. Make linearized predictions
    #
    # NOTE: we need to re-load the model without using BitsAndBytes (our
    # gradient calculations sadly don't currently work with 4/8-bit
    # quantization)
    del model
    t.cuda.empty_cache()
    logging.info("Doing linearized prediction")

    cfg.llm.use_quant = False  # because our gradient calcs don't support bnb
    cfg.llm.use_peft = False  # due to the quirk in loading PEFT models
    # cfg.llm.model_kwargs.attn_implementation = "sdpa"
    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
    model = model.to(device)

    val_loader = dset.loader(
        is_s2s=cfg.llm.is_s2s,
        batch_size=cfg.dset.eval_bs,
        split=cfg.dset.eval_split,
        subset_size=cfg.dset.eval_subset,
    )

    pred_mu = []
    pred_var = []
    pred_logits = []

    metric_kwargs = {"task": "multiclass", "num_classes": dset.n_labels}

    total_loss = 0
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)

    run_baseline = False

    # ---- Baseline

    if run_baseline:

        with t.no_grad():
            for batch in tqdm(val_loader, disable=not cfg.use_tqdm, file=sys.stdout):
                prompts, classes, _ = batch
                classes = classes.to(device)

                batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)

                logits = model(**batch_inputs).logits[:, -1, dset.target_ids.squeeze(-1)]
                # baseline_logits.append(logits.cpu())
                acc_metric(logits, classes)
                ece_metric(logits, classes)
                total_loss += F.cross_entropy(logits, classes).item()

        loss = total_loss / len(val_loader)
        baseline_acc = acc_metric.compute().item()
        baseline_ece = ece_metric.compute().item()

        logging.info(f"Baseline NLL: {loss:.5f}, ACC: {baseline_acc:.5f}, ECE: {baseline_ece:.5f}")
        return 

    def output_callback(outputs: ModelOutput) -> Tensor:
        """Post process model outputs.

        This function will be passed the results of model(**batch_inputs), and
        should return the relevant logits. For multiple-choice tasks, this is
        the class logits, but for full next-token prediction, this would just
        be all the logits.
        """
        # Get the last token for CausalLM
        logits = outputs.logits if cfg.llm.is_s2s else outputs.logits[:, -1]
        # Select the logits corresponding to our target classes
        target_logits = logits[:, dset.target_ids.squeeze(-1)]
        return target_logits

    with t.no_grad():
        for batch in tqdm(val_loader, disable=not cfg.use_tqdm, file=sys.stdout):
            prompts, classes, _ = batch
            classes = classes.to(device)

            batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)

            # Predict the output logit locations
            jacobian, f_mu = jacobian_mean(
                model, batch_inputs, output_callback=output_callback
            )
            pred_mu.append(f_mu.clone().cpu())

            # Predict the output logit variances
            f_var = variance(
                batch_inputs,
                jacobian,
                factors,
                s2,
                dset.n_labels,
                cfg.llm.peft.r,
                cfg.n_kfac,
                device,
            )
            # print(f"f_var shape: {f_var.shape}")

            pred_var.append(f_var.clone().cpu())

            # Sample logits from a Gaussian parametrised by f_mu, f_var
            L = stable_cholesky(f_var)
            samples = 100_000
            f_mu = f_mu.expand(samples, *f_mu.shape)
            L = L.expand(samples, *L.shape)
            eps = t.randn_like(f_mu).unsqueeze(-1)
            logits = f_mu[..., None] + L @ eps

            # this was also a bug - there should be no softmax (?)
            # logits = logits.squeeze(-1).softmax(-1).mean(0)
            logits = logits.squeeze(-1).mean(0)

            pred_logits.append(logits.cpu())
            total_loss += F.cross_entropy(logits, classes).item()
            acc_metric(logits, classes)
            logits = logits.squeeze(-1).mean(0)

            pred_logits.append(logits.cpu())
            total_loss += F.cross_entropy(logits, classes).item()
            acc_metric(logits, classes)
            ece_metric(logits, classes)

    loss = total_loss / len(val_loader)
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()

    # TODO show how this changes with different HPs: lora rank, kronecker
    # TODO how does it compare to the MAP version - does this improve the calibration? 

    logging.info(f"NLL: {loss:.5f}, ACC: {acc:.5f}, ECE: {ece:.5f}")

    output_path = f"{cfg.paths.output_dir}/predicted_logits.pth"
    t.save(
        {"pred_mu": pred_mu, "pred_var": pred_var, "pred_logits": pred_logits},
        output_path,
    )

    print('Validation batches')

    logging.info("Successfully finished.")

    """
Results

Metrics
- NLL = Negative log-likelihood (cross-entropy)
- ACC = Accuracy
- ECE = Expected Calibration Error

Results with different Prior Variance
    Baseline:
    NLL: 0.41716, ACC: 0.86268, ECE: 0.04673  --> Baseline: Finetuned model without Laplace
    NLL: 0.41203, ACC: 0.87324, ECE: 0.05720
    NLL: 0.41238, ACC: 0.87324, ECE: 0.05390
    NLL: 0.41173, ACC: 0.87324, ECE: 0.05046
    NLL: 0.41165, ACC: 0.87324, ECE: 0.05012
    AVG - NLL: 0.4113, ACC: 0.873, ECE: 0.051 (roughly)

    PP 1e-5
    NLL: 0.41119, ACC: 0.87324, ECE: 0.05419

    Prior precision 3e-3:
    NLL: 0.41227, ACC: 0.87324, ECE: 0.04730
    NLL: 0.41154, ACC: 0.87324, ECE: 0.04864

    Prior precision 0.1:
    NLL: 0.41429, ACC: 0.87500, ECE: 0.05723


    # Old with a bug 
    Var 1     -  NLL: nan,     ACC: 0.54049, ECE: 0.22699
    Var 0.1   -  NLL: 1.41531, ACC: 0.79401, ECE: 0.36347 
    Var 0.01  -  NLL: 1.18446, ACC: 0.86444, ECE: 0.15981
    Var 0.001 -  NLL: 1.08173, ACC: 0.86268, ECE: 0.02428  --> The only case with better ECE compared to the baseline, but worse NLL
    Var 3e-4  -  NLL: 1.06210, ACC: 0.86092, ECE: 0.04448
    Var 1e-4  -  NLL: 1.06383, ACC: 0.86444, ECE: 0.04857
    Var 1e-5  -  NLL: 1.06215, ACC: 0.86444, ECE: 0.04143
"""



if __name__ == "__main__":
    main()
