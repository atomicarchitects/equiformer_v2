"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


    Part of code related to auxiliary task is taken from: 
        https://github.com/Open-Catalyst-Project/ocp/pull/335/files
    
"""

import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
#from ocpmodels.trainers.base_trainer import BaseTrainer
from .base_trainer_v2 import BaseTrainerV2, interpolate_init_relaxed_pos
from .engine import AverageMeter


def _mask_input(inputs: torch.Tensor, mask: torch.Tensor):
    masked = inputs[mask]
    return masked


@registry.register_trainer("energy_v2")
class EnergyTrainerV2(BaseTrainerV2):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_.


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
            slurm=slurm,
            noddp=noddp,
        )
        
        # from Graphormer: https://github.com/Open-Catalyst-Project/ocp/pull/335/files
        if self.normalizer.get("normalize_positions", False):
            self.normalizers["positions"] = Normalizer(
                mean=self.normalizer["positions_mean"],
                std=self.normalizer["positions_std"],
                device=self.device,
            )
        # variables related to auxiliary_task_loss scheduling
        if self.use_auxiliary_task:
            self.total_steps = len(self.train_loader) * self.config["optim"]["max_epochs"]
            self.current_auxiliary_task_weight = self.auxiliary_task_weight
            

    def load_task(self):
        #logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.file_logger.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    @torch.no_grad()
    def predict(
        self, loader, per_image=True, results_file=None, disable_tqdm=False
    ):
        #if distutils.is_master() and not disable_tqdm:
        #    logging.info("Predicting on test.")
        self.file_logger.info("Predicting on test.")
        
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        predictions = {"id": [], "energy": []}
        pos_preds = {}  # save predicted relaxed position

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )

            if per_image:
                predictions["id"].extend(
                    [str(i) for i in batch[0].sid.tolist()]
                )
                predictions["energy"].extend(out["energy"].tolist())
            else:
                predictions["energy"] = out["energy"].detach()
                return predictions
            
            if self.config["task"].get("write_pos", False):
                assert 'pos' in out.keys()
                delta_pos = out['pos']
                
                if self.normalizer.get("normalize_positions", False):
                    delta_pos = self.normalizers["positions"].denorm(delta_pos)
                    
                # mask out fixed atoms
                tag_mask = batch[0].tags.to(self.device)
                tag_mask = (tag_mask > 0)
                pred_pos = batch[0].pos.to(self.device)
                pred_pos[tag_mask] = pred_pos[tag_mask] + delta_pos[tag_mask]
                
                natoms = batch[0].natoms.tolist()
                pred_pos_list = torch.split(pred_pos, natoms)
                sid_list = [str(sid) for sid in batch[0].sid.tolist()]
                for j in range(len(sid_list)):
                    pos_preds[sid_list[j]] = pred_pos_list[j].detach().cpu()                
                
        self.save_results(predictions, results_file, keys=["energy"])
        
        if self.config["task"].get("write_pos", False):
            torch.save(pos_preds, 
                os.path.join(self.run_dir, 'pos_pred_{}.pt'.format(distutils.get_rank())))
            distutils.synchronize()
            if distutils.is_master():
                gather_pos_preds = {}
                for i in range(distutils.get_world_size()):
                    rank_pos_preds = torch.load(
                        os.path.join(self.run_dir, 'pos_pred_{}.pt'.format(i)))
                    for k, v in rank_pos_preds.items():
                        if k not in gather_pos_preds.keys():
                            gather_pos_preds[k] = v
                torch.save(gather_pos_preds, os.path.join(self.run_dir, 'pos_pred.pt'))

        if self.ema:
            self.ema.restore()

        return predictions

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_mae = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)
            
            avg_metric_dict = {}
            for k in self.evaluator.metric_fn:
                avg_metric_dict[k] = AverageMeter()
            avg_metric_dict['loss'] = AverageMeter()

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)
                
                # Interpolate between initial and relaxed pos
                if self.use_interpolate_init_relaxed_pos:
                    batch = [interpolate_init_relaxed_pos(batch_data) for batch_data in batch]

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                if self.grad_accumulation_steps != 1:
                    loss = loss / self.grad_accumulation_steps
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale * self.grad_accumulation_steps, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                for k, v in log_dict.items():
                    avg_metric_dict[k].update(v)
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    (self.step % self.config["cmd"]["print_every"] == 0
                     or i == 0
                     or i == (len(self.train_loader) - 1))
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    #log_str = [
                    #    "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    #]
                    #print(", ".join(log_str))
                    log_str = []
                    for k, v in log_dict.items():
                        temp = "{}: {:.2e}".format(k, v)
                        if k in avg_metric_dict.keys():
                            temp = temp + " ({:.2e})".format(avg_metric_dict[k].avg)
                        log_str.append(temp)
                    log_str = ", ".join(log_str)
                    if self.use_auxiliary_task:
                        log_str = log_str + ", aux_weight: {:.2e}".format(
                            self.current_auxiliary_task_weight)
                    self.file_logger.info(log_str)
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                # Evaluate on val set after every `eval_every` iterations.
                if (self.step % eval_every == 0
                    or i == (len(self.train_loader) - 1)):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                            use_ema=False
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=True,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )
                        # evaluate with model EMA
                        if self.ema is not None:
                            val_ema_metrics = self.validate(split="val",
                                disable_tqdm=disable_eval_tqdm, use_ema=True)
                            if (val_ema_metrics[self.evaluator.task_primary_metric[self.name]]["metric"] < self.best_val_mae):
                                self.best_val_mae = val_metrics[self.evaluator.task_primary_metric[self.name]]["metric"]
                                self.save(metrics=val_metrics,
                                    checkpoint_file="best_ema_checkpoint.pt",
                                    training_state=False)

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    if self.grad_accumulation_steps != 1:
                        if self.step % self.grad_accumulation_steps == 0:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()

            torch.cuda.empty_cache()

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        output = self.model(batch_list)
        
        if not self.use_auxiliary_task:
            if output.shape[-1] == 1:
                output = output.view(-1)
            return {'energy': output}
        else: # IS2RE + IS2RS
            pass
            output_energy, output_pos = output[0], output[1]
            if output_energy.shape[-1] == 1:
                output_energy = output_energy.view(-1)
            return {'energy': output_energy, 'pos': output_pos}
        

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.loss_fn["energy"](out["energy"], target_normed)
        
        if self.use_auxiliary_task:
            pos = torch.cat([batch.pos.to(self.device) for batch in batch_list], dim=0)
            pos_relaxed = torch.cat([batch.pos_relaxed.to(self.device) for batch in batch_list], dim=0)
            delta_pos = pos_relaxed - pos
            # normalize delta_pos
            # for 1e, we divide by L2-norm only
            if self.normalizer.get("normalize_positions", False):
                delta_pos = self.normalizers["positions"].norm(delta_pos)
            # mask out fixed atoms
            tag_mask = torch.cat([batch.tags.to(self.device) for batch in batch_list], dim=0)
            tag_mask = (tag_mask > 0)
            
            loss_aux = self.loss_fn['force'](_mask_input(out['pos'], tag_mask), 
                _mask_input(delta_pos, tag_mask))
            self._compute_auxiliary_task_weight()
            loss = loss + loss_aux * self.current_auxiliary_task_weight
            
        return loss
    

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        return metrics
    
    
    def _compute_auxiliary_task_weight(self):
        # linearly decay self.auxiliary_task_weight to 1 
        # throughout the whole training procedure
        _min_weight = 1
        weight = self.auxiliary_task_weight
        weight_range = max(0.0, weight - _min_weight)
        weight = weight - weight_range * min(1.0, ((self.step + 0.0) / self.total_steps))
        self.current_auxiliary_task_weight = weight
        return