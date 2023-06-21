"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import json
import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
import torch_geometric
from torch_cluster import radius_graph
from tqdm import tqdm

import ocpmodels
from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    plot_histogram,
    save_checkpoint,
    warmup_lr_lambda,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler as LRSchedulerOC20

from .base_trainer_oc20 import BaseTrainer
from .logger import FileLogger
from .lr_scheduler import LRScheduler
from .engine import AverageMeter


def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    name_no_wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name
            or any(name.endswith(skip_name) for skip_name in skip_list)):
            no_decay.append(param)
            name_no_wd.append(name)
        else:
            decay.append(param)
    name_no_wd.sort()
    params = [{'params': no_decay, 'weight_decay': 0.},
              {'params': decay, 'weight_decay': weight_decay}]
    return params, name_no_wd


def interpolate_init_relaxed_pos(batch):
    _interpolate_threshold = 0.5
    _min_interpolate_factor = 0.0 #0.1
    _gaussian_noise_std = 0.3

    batch_index = batch.batch
    batch_size = batch_index.max() + 1

    threshold_tensor = torch.rand((batch_size, 1), dtype=batch.pos.dtype, device=batch.pos.device)
    threshold_tensor = threshold_tensor + (1 - _interpolate_threshold)
    threshold_tensor = threshold_tensor.floor_() # 1: has interpolation, 0: no interpolation
    threshold_tensor = threshold_tensor[batch_index]

    interpolate_factor = torch.zeros((batch_index.shape[0], 1),
        dtype=batch.pos.dtype, device=batch.pos.device)
    interpolate_factor = interpolate_factor.uniform_(_min_interpolate_factor, 1)

    noise_vec = torch.zeros((batch_index.shape[0], 3),
        dtype=batch.pos.dtype, device=batch.pos.device)
    noise_vec = noise_vec.uniform_(-1, 1)
    noise_vec_norm = noise_vec.norm(dim=1, keepdim=True)
    noise_vec = noise_vec / (noise_vec_norm + 1e-6)
    noise_scale = torch.zeros((batch_index.shape[0], 1),
        dtype=batch.pos.dtype, device=batch.pos.device)
    noise_scale = noise_scale.normal_(mean=0, std=_gaussian_noise_std)
    noise_vec = noise_vec * noise_scale

    noise_vec = noise_vec.normal_(mean=0, std=_gaussian_noise_std)

    #interpolate_factor = interpolate_factor * threshold_tensor
    #interpolate_factor = 1 - interpolate_factor
    #assert torch.all(interpolate_factor >= 0.0)
    #assert torch.all(interpolate_factor <= 1.0)
    #interpolate_factor = interpolate_factor[batch_index]
    #batch.pos = batch.pos * interpolate_factor + (1 - interpolate_factor) * batch.pos_relaxed

    tags = batch.tags
    tags = (tags > 0)
    pos = batch.pos
    pos_relaxed = batch.pos_relaxed
    pos_interpolated = pos * interpolate_factor + (1 - interpolate_factor) * pos_relaxed
    pos_noise = pos_interpolated + noise_vec
    new_pos = pos_noise * threshold_tensor + pos * (1 - threshold_tensor)
    batch.pos[tags] = new_pos[tags]

    return batch


'''
    1. Inherit from `BaseTrainer` in `base_trainer_oc20.py` and remove redundant parts.
    2. Use PyTorch lr scheduler and use LambdaLR for consine and multi-step lr scheduling.
    3. Add no weight decay.
    4. Add auxiliary task (IS2RE + IS2RS).
'''
@registry.register_trainer("base_v2")
class BaseTrainerV2(BaseTrainer):
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
        name="base_trainer",
        slurm={},
        noddp=False,
    ):
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available
        if run_dir is None:
            run_dir = os.getcwd()
        self.run_dir = run_dir

        if timestamp_id is None:
            timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
                self.device
            )
            # create directories from master rank only
            distutils.broadcast(timestamp, 0)
            timestamp = datetime.datetime.fromtimestamp(
                timestamp.int()
            ).strftime("%Y-%m-%d-%H-%M-%S")
            if identifier:
                self.timestamp_id = f"{timestamp}-{identifier}"
            else:
                self.timestamp_id = timestamp
        else:
            self.timestamp_id = timestamp_id

        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        ocpmodels.__path__[0],
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "noddp": noddp,
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", None)
            self.config["val_dataset"] = dataset.get("val", None)
            self.config["test_dataset"] = dataset.get("test", None)
        else:
            self.config["dataset"] = dataset

        self.normalizer = normalizer
        # This supports the legacy way of providing norm parameters in dataset
        if self.config.get("dataset", None) is not None and normalizer is None:
            self.normalizer = self.config["dataset"]

        if not is_debug and distutils.is_master() and not is_hpo:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_hpo = is_hpo

        if self.is_hpo:
            # conditional import is necessary for checkpointing
            from ray import tune

            from ocpmodels.common.hpo_utils import tune_reporter

            # sets the hpo checkpoint frequency
            # default is no checkpointing
            self.hpo_checkpoint_every = self.config["optim"].get(
                "checkpoint_every", -1
            )

        #if distutils.is_master():
        #    print(yaml.dump(self.config, default_flow_style=False))
        self.file_logger = FileLogger(is_master=distutils.is_master(),
            is_rank0=distutils.is_master(), output_dir=run_dir)
        self.file_logger.info(yaml.dump(self.config, default_flow_style=False))

        # auxiliary task
        self.auxiliary_task_weight = self.config['optim'].get('auxiliary_task_weight', 0.0)
        self.use_auxiliary_task = False
        if self.auxiliary_task_weight > 0.0:
            self.use_auxiliary_task = True
            #self.config['model_attributes']['use_auxiliary_task'] = True
            #self.file_logger.info('Use auxiliary task and modify `model_attributes`.')

        # for interpolating initial pos and relaxed pos
        self.use_interpolate_init_relaxed_pos = self.config['optim'].get('use_interpolate_init_relaxed_pos', False)

        # gradient accumulation
        # based on https://github.com/microsoft/Swin-Transformer/blob/main/main.py
        self.grad_accumulation_steps = self.config['optim'].get('grad_accumulation_steps', 1)

        self.load()

        self.evaluator = Evaluator(task=name)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()


    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            raise ValueError

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def load_model(self):
        # Build model
        #if distutils.is_master():
        #    logging.info(f"Loading model: {self.config['model']}")
        self.file_logger.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        self.model = registry.get_model_class(self.config["model"])(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        # for no weight decay
        self.model_params_no_wd = {}
        if hasattr(self.model, 'no_weight_decay'):
            self.model_params_no_wd = self.model.no_weight_decay()

        #if distutils.is_master():
        #    logging.info(
        #        f"Loaded {self.model.__class__.__name__} with "
        #        f"{self.model.num_params} parameters."
        #    )
        self.file_logger.info(self.model)
        self.file_logger.info(
            f"Loaded {self.model.__class__.__name__} with "
            f"{self.model.num_params} parameters."
        )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )


    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)
        optimizer_params = self.config['optim']['optimizer_params']
        weight_decay = optimizer_params['weight_decay']

        parameters, name_no_wd = add_weight_decay(self.model,
            weight_decay, self.model_params_no_wd)
        self.file_logger.info('Parameters without weight decay:')
        self.file_logger.info(name_no_wd)

        self.optimizer = optimizer(
            parameters,
            lr=self.config["optim"]["lr_initial"],
            **optimizer_params,
        )

        '''
        if self.config["optim"].get("weight_decay", 0) > 0:

            # Do not regularize bias etc.
            params_decay = []
            params_no_decay = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        params_no_decay += [param]
                    elif "frequencies" in name:
                        params_no_decay += [param]
                    elif "bias" in name:
                        params_no_decay += [param]
                    else:
                        params_decay += [param]

            self.optimizer = optimizer(
                [
                    {"params": params_no_decay, "weight_decay": 0},
                    {
                        "params": params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        '''


    def load_extras(self):

        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]['scheduler_params']['epochs'] = self.config["optim"]["max_epochs"]
        self.config["optim"]['scheduler_params']['lr'] = self.config["optim"]["lr_initial"]

        # convert epochs into number of steps
        n_iter_per_epoch = len(self.train_loader)
        if self.grad_accumulation_steps != 1:
            n_iter_per_epoch = n_iter_per_epoch // self.grad_accumulation_steps
        scheduler_params = self.config['optim']['scheduler_params']
        for k in scheduler_params.keys():
            if 'epochs' in k:
                if isinstance(scheduler_params[k], (int, float, list)):
                    scheduler_params[k] = multiply(scheduler_params[k], n_iter_per_epoch)

        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None


    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, use_ema=True):

        self.file_logger.info(f"Evaluating on {split}.")

        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema and use_ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        log_str = ", ".join(log_str)
        log_str = "[{}] ".format(split) + log_str
        self.file_logger.info(log_str)

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema and use_ema:
            self.ema.restore()

        return metrics

    def _backward(self, loss):
        if self.grad_accumulation_steps == 1:
            self.optimizer.zero_grad()

        loss.backward()

        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )

        if (self.grad_accumulation_steps != 1):
            if (self.step % self.grad_accumulation_steps != 0):
                return

        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

        if (self.grad_accumulation_steps != 1):
            if (self.step % self.grad_accumulation_steps == 0):
                self.optimizer.zero_grad()


    def compute_stats(self):
        '''
            Compute mean of numbers of nodes and edges

            Assume using cpu
        '''
        self._otf_graph = True
        self._use_pbc = True
        self._max_radius = 8.0
        self._max_neighbors = 40
        log_str = '\nCalculating statistics with '
        log_str = log_str + 'otf_graph={}, use_pbc={}, max_radius={}, max_neighbors={}\n'.format(
            self._otf_graph, self._use_pbc, self._max_radius, self._max_neighbors)
        self.file_logger.info(log_str)

        avg_node = AverageMeter()
        avg_edge = AverageMeter()
        avg_degree = AverageMeter()
        avg_delta_pos_l2_norm = AverageMeter()
        for i, batch_list in enumerate(self.train_loader):
            data = batch_list[0]

            if self.use_interpolate_init_relaxed_pos:
                data = interpolate_init_relaxed_pos(data)

            data = self._forward_otf_graph(data)
            edge_index, edge_vec, edge_length, offsets = self._forward_use_pbc(data)

            batch = data.batch
            batch_size = float(batch.max() + 1)
            num_nodes = data.pos.shape[0]
            edge_src = edge_index[0]
            num_edges = edge_src.shape[0]
            num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
            num_degree = torch.sum(num_degree)

            delta_pos = data.pos_relaxed - data.pos
            tag_mask = data.tags
            tag_mask = (tag_mask > 0)
            delta_pos = self._mask_input(delta_pos, tag_mask)
            delta_pos_norm = torch.sum(delta_pos.pow(2), dim=-1)
            delta_pos_norm = delta_pos_norm.pow(0.5)
            delta_pos_norm = torch.sum(delta_pos_norm)

            avg_node.update(num_nodes / batch_size, batch_size)
            avg_edge.update(num_edges / batch_size, batch_size)
            avg_degree.update(num_degree / (num_nodes), num_nodes)
            avg_delta_pos_l2_norm.update(delta_pos_norm / delta_pos.shape[0], delta_pos.shape[0])

            if i % self.config["cmd"]["print_every"] == 0 or i == (len(self.train_loader) - 1):
                log_str = '[{}/{}]\tavg node: {}, '.format(i, len(self.train_loader), avg_node.avg)
                log_str += 'avg edge: {}, '.format(avg_edge.avg)
                log_str += 'avg degree: {}, '.format(avg_degree.avg)
                log_str += 'avg delta pos l2 norm: {}'.format(avg_delta_pos_l2_norm.avg)
                self.file_logger.info(log_str)


    def _forward_otf_graph(self, data):
        if self._otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self._max_radius, self._max_neighbors
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
            return data
        else:
            return data


    def _forward_use_pbc(self, data):
        pos = data.pos
        batch = data.batch
        if self._use_pbc:
            out = get_pbc_distances(pos,
                data.edge_index,
                data.cell, data.cell_offsets,
                data.neighbors,
                return_offsets=True)
            edge_index = out["edge_index"]
            offsets = out["offsets"]
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst) + offsets
            dist = edge_vec.norm(dim=1)
        else:
            edge_index = radius_graph(pos, r=self._max_radius,
                batch=batch, max_num_neighbors=self._max_neighbors)
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
            dist = edge_vec.norm(dim=1)
            offsets = None
        return edge_index, edge_vec, dist, offsets


    def _mask_input(self, inputs, mask):
        return inputs[mask]
