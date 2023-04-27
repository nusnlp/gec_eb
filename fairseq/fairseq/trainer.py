# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
from itertools import chain
import logging
import math
import os
import sys
from typing import Any, Dict, List

import torch
import numpy
import torch.nn as nn

from fairseq import checkpoint_utils, distributed_utils, metrics, models, optim, utils, progress_bar
from fairseq.file_io import PathManager
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.models import BaseActor, AveEmbActor, LanguageActor, LSTMActor, PPLFeatureActor, EmbeddingActor1, EmbeddingActor2
from fairseq.data import encoders


logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self._criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        if self.cuda:
            self._criterion = self._criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None

        if self.cuda and args.distributed_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(args.distributed_world_size)
        else:
            self._grad_norm_buf = None
        self.nos_temperature = None
        self.ori_temperature = None
        self.data_actor_ori = None
        self.data_actor_nos = None
        self.data_optimizer_ori = None
        self.data_optimizer_nos = None
        self.data_actor_count = 0
        self.update_index_list = []
        self.ori_weight_list = []
        self.nos_weight_list = []
        self.nos_all_weight_list = []
        self.baseline = None
        metrics.log_start_time("wall", priority=790, round=0)
        self._weights = None
        self._w_decay = None
        self.epsilon = None
        self.number = 0
        self.feature = None
        self.ori_weight_trainer = None
        self.pretrain = True
        self.rl_update = 0

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )

        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.args, params
                )
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, params)

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state["metrics"] = metrics.state_dict()
            checkpoint_utils.save_state(
                filename,
                self.args,
                self.get_model().state_dict(),
                self.get_criterion(),
                self.optimizer,
                self.lr_scheduler,
                self.get_num_updates(),
                self._optim_history,
                extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        bexists = PathManager.isfile(filename)
        if bexists:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(
                    state["model"], strict=True, args=self.args
                )
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]
            last_optim_state = state.get("last_optimizer_state", None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            epoch = extra_state["train_iterator"]["epoch"]
            logger.info(
                "loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

            self.lr_step(epoch)

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, TimeMeter):
                        meter.reset()
        else:
            logger.info("no existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
            )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.args.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size if shard_batch_itr else 1,
            shard_id=self.args.distributed_rank if shard_batch_itr else 0,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def init_data_actor(self):
        if self.args.data_actor == 'ppl_feature':
            self.data_actor = PPLFeatureActor(self.args)
            if self.cuda:
                self.data_actor = self.data_actor.cuda()
            self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
        elif self.args.data_actor == 'base':
            self.data_actor = BaseActor(self.args, 2)
            # if self.args.pretrained_data_actor_path is not None:
            #     logger.info("Loading pretrained weight from {}".format(self.args.pretrained_data_actor_path))
            #     self.data_actor.load_state_dict(torch.load(self.args.pretrained_data_actor_path))
            if self.cuda:
                self.data_actor = self.data_actor.cuda()
            self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad],
                                                   lr=self.args.data_actor_lr)
        elif self.args.data_actor == 'embedding':
            self.nos_temperature = self.args.nos_temperature
            self.ori_temperature = self.args.ori_temperature
            self.ori_weight_trainer = self.args.ori_weight_trainer
            self.number = self.args.num_sources - 2
            self.feature = torch.tensor([i for i in range(self.number)])
            self.data_actor_ori = EmbeddingActor1(2)
            self.data_actor_nos = EmbeddingActor2(self.number)
            self.data_optimizer_ori = torch.optim.Adam([p for p in self.data_actor_ori.parameters() if p.requires_grad],
                                                       lr=self.args.data_actor_lr)
            self.data_optimizer_nos = torch.optim.Adam([p for p in self.data_actor_nos.parameters() if p.requires_grad],
                                                       lr=self.args.nos_data_actor_lr)

            for i in range(self.number):
                self.nos_all_weight_list.append([])
            # print(self.nos_all_weight_list)
            # print(self.number)

            if self.args.pretrained_data_actor_path is not None:
                logger.info("Loading pretrained weight from {}".format(self.args.pretrain_data_actor_ori))
                ori_ckpt = torch.load(self.args.pretrain_data_actor_ori)
                self.data_actor_ori.load_state_dict(ori_ckpt["net"])
                self.data_optimizer_ori.load_state_dict(ori_ckpt["optimizer"])
                self.data_optimizer_ori.param_groups[0]['lr'] = self.args.data_actor_lr
                for state in self.data_optimizer_ori.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                logger.info("Loading pretrained weight from {}".format(self.args.pretrain_data_actor_nos))
                nos_ckpt = torch.load(self.args.pretrain_data_actor_nos)
                self.data_actor_nos.load_state_dict(nos_ckpt["net"])
                self.data_optimizer_nos.load_state_dict(nos_ckpt["optimizer"])
                self.data_optimizer_nos.param_groups[0]['lr'] = self.args.nos_data_actor_lr
                for state in self.data_optimizer_nos.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            if self.cuda:
                self.data_actor_ori = self.data_actor_ori.cuda()
                self.data_actor_nos = self.data_actor_nos.cuda()

    def compute_valid_grad(self, task, args):
        self.model.train()
        self.criterion.train()
        self.zero_grad()
        self.optimizer.clear_cache()
        # logger.info("Before computing the validation gradient")
        # logger.info(self.optimizer.print_state())
        # TODO: get the reward for N batches
        # TODO: 1. calculate the dev set gradient
        itr = task.get_batch_iterator(
            dataset=task.dataset('valid'),
            max_tokens=self.number * self.args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                self.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        tmp_sample_size = 0
        for val_sample in itr:
            val_sample['valid'] = ""
            del val_sample['net_input']['prev_output_tokens_neg_padded']
            val_sample = self._prepare_sample(val_sample)
            if val_sample is None:
                val_sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False
            tmp_loss, tmp_sample_size_i, tmp_logging_output = self.task.train_step(
                val_sample, self.model, self.criterion, self.optimizer, ignore_grad)
            if not ignore_grad:
                tmp_sample_size += tmp_sample_size_i
        if tmp_sample_size > 0:
            if self._sync_stats():
                self.optimizer.multiply_grads(self.args.distributed_world_size / tmp_sample_size)
            else:
                self.optimizer.multiply_grads(1 / tmp_sample_size)
        self.optimizer.save_dev_grad()
        self.zero_grad()
        # logger.info(self.optimizer.print_state())
        # logger.info("validation gradient saved")

    def compute_weight(self, samples, ori_weight=None, nos_weight=None, raise_oom=False):
        # TODO: 2. calculate the train set gradient (in this method, don't need to save)
        # logger.info("Before calculating the reward")
        sample_size_list = []
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):
            # print(i)
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                        self.args.distributed_world_size > 1
                        and hasattr(self.model, "no_sync")
                        and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                # with maybe_no_sync():
                # forward and backward
                loss, sample_size_i, logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer, ignore_grad, ori_prob=ori_weight,
                    nos_prob=nos_weight, backward=False
                )
                # logger.info("Returned loss from compute_weight(): {}".format(loss))
                # del loss
                # total_loss_list.append(loss)
                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    # sample_size += sample_size_i
                else:
                    sample_size_i = 0
                sample_size_list.append(sample_size_i)
                sim_list_nos = []
                sim_list_ori = []
                # TODO: 3. calculate the reward
                for noise_idx in range(self.number):
                    self.optimizer.backward(loss[noise_idx], retain_graph=True)
                    # self.optimizer.backward(loss[noise_idx])
                    if sample_size_i > 0:
                        if self._sync_stats():
                            self.optimizer.multiply_grads(self.args.distributed_world_size / sample_size_i)
                        else:
                            self.optimizer.multiply_grads(1 / sample_size_i)
                    nos_sim, cur_grad_sim, prev_grad_sim = self.optimizer.get_grad_sim()
                    # logger.info("{}th reward is computed".format(noise_idx))
                    sim_list_nos.append(nos_sim)
                    self.zero_grad()
                for noise_idx in ["ori", "nos"]:
                    if noise_idx == "ori":
                        idx = self.number
                        target_loss = loss[idx]
                        self.optimizer.backward(target_loss)
                    else:
                        idx = self.number + 1
                        target_loss = loss[idx]
                        self.optimizer.backward(target_loss)
                        # self.optimizer.backward(target_loss, retain_graph=True)
                    # self.optimizer.backward(target_loss)
                    if sample_size_i > 0:
                        if self._sync_stats():
                            self.optimizer.multiply_grads(self.args.distributed_world_size / sample_size_i)
                        else:
                            self.optimizer.multiply_grads(1 / sample_size_i)
                    ori_sim, cur_grad_sim, prev_grad_sim = self.optimizer.get_grad_sim()
                    # logger.info("{} gradient is computed".format(noise_idx))
                    sim_list_ori.append(ori_sim)
                    self.zero_grad()
                # TODO: 4. update two data actors based on the reward
                # TODO: update phi_1
                if self.args.normalize:
                    np_arr = numpy.array(sim_list_ori)
                    normed = (np_arr - np_arr.mean(axis=0)) / (np_arr.std(axis=0) + 1e-8)
                    grad_scale_ori = torch.from_numpy(normed).view(1, -1)
                    grad_scale_ori = grad_scale_ori.type(torch.FloatTensor)
                else:
                    grad_scale_ori = torch.FloatTensor(sim_list_ori).view(1, -1)
                if self.cuda:
                    grad_scale_ori = grad_scale_ori.cuda()
                for _ in range(self.args.data_actor_optim_step):
                    a_logits = self.data_actor_ori.forward(sample, just_score=True)
                    loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                    loss = (loss * grad_scale_ori).sum()
                    loss.backward()
                    self.data_optimizer_ori.step()
                    self.data_optimizer_ori.zero_grad()
                # TODO: update phi_2
                if self.args.normalize:
                    np_arr = numpy.array(sim_list_nos)
                    normed = (np_arr - np_arr.mean(axis=0)) / (np_arr.std(axis=0) + 1e-8)
                    grad_scale_nos = torch.from_numpy(normed).view(1, -1)
                    grad_scale_nos = grad_scale_nos.type(torch.FloatTensor)
                else:
                    grad_scale_nos = torch.FloatTensor(sim_list_nos).view(1, -1)
                if self.cuda:
                    grad_scale_nos = grad_scale_nos.cuda()
                for _ in range(self.args.data_actor_optim_step):
                    a_logits = self.data_actor_nos.forward(self.feature, just_score=True)
                    loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                    loss = (loss * grad_scale_nos).sum()
                    loss.backward()
                    self.data_optimizer_nos.step()
                    self.data_optimizer_nos.zero_grad()
                    # emptying the CUDA cache after the first step can
                    # reduce the chance of OOM
                # if self.cuda and self.get_num_updates() == 0:
                #     torch.cuda.empty_cache()
                # logger.info("reward for ori {}, nos {}".format(grad_scale_ori, grad_scale_nos))
                # logger.info("phi1 and phi2 has been updated!!!")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

            if self.args.plot:
                with torch.no_grad():
                    a = []
                    ori_output = self.data_actor_ori.forward(a, just_score=True)[0]
                    nos_output = self.data_actor_nos.forward(self.feature, just_score=True)[0]
                    tmp_ori_prob = torch.nn.functional.softmax(ori_output, dim=-1)
                    tmp_nos_prob = torch.nn.functional.softmax(nos_output, dim=-1)

                    self.update_index_list.append(self.get_num_rl_updates())
                    self.ori_weight_list.append(tmp_ori_prob[0].item())
                    self.nos_weight_list = [tmp_ori_prob[1].item()]

                    for i in range(self.number):
                        self.nos_all_weight_list[i].append(tmp_nos_prob[i].item())
                    # self.optimizer.clear_cache()
        if self.args.pretrain:
            self.set_num_updates(self.get_num_updates() + 1)
            logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
            return logging_output
        else:
            return None

    def get_num_rl_updates(self):
        self.rl_update += 1
        return self.rl_update

    def get_weight(self):
        a = []
        with torch.no_grad():
            ori_output = self.data_actor_ori.forward(a, just_score=True)[0]
            nos_output = self.data_actor_nos.forward(self.feature, just_score=True)[0]
            tmp_ori_prob = torch.nn.functional.softmax(ori_output, dim=-1)
            tmp_nos_prob = torch.nn.functional.softmax(nos_output, dim=-1)
            #
            # self.update_index_list.append(self.get_num_rl_updates())
            # self.ori_weight_list.append(tmp_ori_prob[0].item())
            # self.nos_weight_list = [tmp_ori_prob[1].item()]
            #
            # for i in range(self.number):
            #     self.nos_all_weight_list[i].append(tmp_nos_prob[i].item())
            # # self.optimizer.clear_cache()
        return tmp_ori_prob, tmp_nos_prob

    def save_data_actor(self, epoch):
        save_path_ori = os.path.join(self.args.save_dir, "data_actor.ori." + str(epoch) + ".ckpt")
        logger.info("Saving the data actor to {}".format(save_path_ori))
        ori_checkpoint = {
            "net": self.data_actor_ori.state_dict(),
            "optimizer": self.data_optimizer_ori.state_dict(),
        }
        torch.save(ori_checkpoint, save_path_ori)
        save_path_nos = os.path.join(self.args.save_dir, "data_actor.nos." + str(epoch) + ".ckpt")
        logger.info("Saving the data actor to {}".format(save_path_nos))
        nos_checkpoint = {
            "net": self.data_actor_nos.state_dict(),
            "optimizer": self.data_optimizer_nos.state_dict(),
        }
        torch.save(nos_checkpoint, save_path_nos)

    def save_csv_file(self):
        import csv
        csv_name = "lr=" + str(self.args.data_actor_lr) + "update_frq=" + str(
            self.args.update_freq) + "max_tokens=" + str(self.args.max_tokens) + "data=" + str(
            self.args.data.split("/")[-1]) + ".csv"
        csv_path = os.path.join(self.args.csv_path, csv_name)
        writer = csv.writer(open(csv_path, "w+"))
        writer.writerow(self.update_index_list)
        writer.writerow(self.ori_weight_list)
        writer.writerow(self.nos_weight_list)
        for i in range(self.number):
            writer.writerow(self.nos_all_weight_list[i])

    @metrics.aggregate("train")
    def train_step(self, samples, ori_prob, nos_prob, dummy_batch=False, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch is None:
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            metrics.log_start_time("train_wall", priority=800, round=0)

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):
            if "reweight_modified" in self.args.criterion:
                sample['test'] = None
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    # logger.info("ori_prob content : {}, nos_prob content: {}".format(ori_prob, nos_prob))
                    loss, sample_size_i, logging_output = self.task.train_step(sample, self.model, self.criterion,
                                                                               self.optimizer, ignore_grad,
                                                                               ori_prob=ori_prob,
                                                                               nos_prob=nos_prob)
                    # logger.info("Returned loss from train_step(): {}".format(loss))
                    del loss

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self._sync_stats():
            logging_outputs, sample_size, ooms = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms,
            )

        metrics.log_scalar("oom", ooms, len(samples), priority=600, round=3)
        if ooms == self.args.distributed_world_size * len(samples):
            logger.warning("OOM in all workers, skipping update")
            self.zero_grad()
            return None

        try:
            # normalize grads by sample size
            if sample_size > 0:
                if self._sync_stats():
                    # multiply gradients by (# GPUs / sample_size) since DDP
                    # already normalizes by the number of GPUs. Thus we get
                    # (sum_of_gradients / sample_size).
                    self.optimizer.multiply_grads(self.args.distributed_world_size / sample_size)
                else:
                    self.optimizer.multiply_grads(1 / sample_size)

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)

            # check that grad norms are consistent across workers
            if not self.args.use_bmuf:
                self._check_grad_norms(grad_norm)

            # take an optimization step
            self.optimizer.step()
            self.set_num_updates(self.get_num_updates() + 1)

            # task specific update per step
            self.task.update_step(self.get_num_updates())

            # log stats
            logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
            metrics.log_speed("ups", 1., priority=100, round=2)
            metrics.log_scalar("gnorm", utils.item(grad_norm), priority=400, round=3)
            metrics.log_scalar(
                "clip",
                100 if grad_norm > self.args.clip_norm > 0 else 0,
                priority=500,
                round=1,
            )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.args.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.args.empty_cache_freq - 1)
                    % self.args.empty_cache_freq
                ) == 0
                and torch.cuda.is_available()
                and not self.args.cpu
            ):
                torch.cuda.empty_cache()
        except OverflowError as e:
            logger.info("NOTE: overflow detected, " + str(e))
            self.zero_grad()
            logging_output = None
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        if self.args.fp16:
            metrics.log_scalar("loss_scale", self.optimizer.scaler.loss_scale, priority=700, round=0)

        metrics.log_stop_time("train_wall")

        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            sample["valid"] = None
            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            if ignore_results:
                logging_outputs, sample_size = [], 0
            else:
                logging_outputs = [logging_output]

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_size = self._aggregate_logging_outputs(
                logging_outputs, sample_size
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from fairseq import meters

        if 'get_meter' not in self._warn_once:
            self._warn_once.add('get_meter')
            utils.deprecation_warning(
                'Trainer.get_meter is deprecated. Please use fairseq.metrics instead.'
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_"):]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        return self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
                and (self.get_num_updates() + 1) > self.args.warmup_iterations
            )
        )

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum
    ):
        if self.get_criterion().__class__.logging_outputs_can_be_summed():
            return self._fast_stat_sync_sum(logging_outputs, *extra_stats_to_sum)
        else:
            return self._all_gather_list_sync(logging_outputs, *extra_stats_to_sum)

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384),
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return [logging_outputs] + extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        min_buffer_size: int = 50,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed.
        """
        num_extra = len(extra_stats_to_sum)
        if len(logging_outputs) > 0:
            sorted_keys = sorted(logging_outputs[0].keys())
            stats = [0.] + list(extra_stats_to_sum) + [
                sum(log.get(k, 0) for log in logging_outputs)
                for k in sorted_keys
            ]
            stats = stats + [0.]*(min_buffer_size - len(stats))
            buf = torch.cuda.DoubleTensor(stats)
        else:
            buf = torch.zeros(min_buffer_size, dtype=torch.double, device='cuda')
            buf[0] = 1.  # flag to indicate we should fallback to _all_gather_list_sync

        # stats buffer is organized like:
        # 0: flag to indicate whether fast-stat-sync should be disabled
        # 1-i: extra_stats_to_sum
        # i-j: values from logging_outputs (sorted by key)
        # j-min_buffer_size: padded with 0s
        distributed_utils.all_reduce(buf)

        buf = buf.tolist()
        fallback = buf[0]
        if fallback > 0.:
            # fallback to _all_gather_list_sync
            return self._all_gather_list_sync(logging_outputs, *extra_stats_to_sum)
        else:
            extra_stats_to_sum, stats = buf[1:num_extra + 1], buf[num_extra + 1:]
            stats = [{k: stats[i] for i, k in enumerate(sorted_keys)}]
            return [stats] + extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.args.distributed_rank] = grad_norm
            distributed_utils.all_reduce(self._grad_norm_buf)
            if not (self._grad_norm_buf == self._grad_norm_buf[0]).all():
                raise RuntimeError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=no_c10d."
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size):
        with metrics.aggregate() as agg:
            # convert logging_outputs to CPU to avoid unnecessary
            # device-to-host transfers in reduce_metrics
            logging_outputs = utils.apply_to_sample(
                lambda t: t.to(device='cpu', non_blocking=True),
                logging_outputs
            )

            self.task.reduce_metrics(logging_outputs, self.get_criterion())

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output
