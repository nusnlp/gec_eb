# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math

class FairseqOptimizer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    # TODO: CHN: newly copied function
    def save_dev_grad_multi(self, utility='ave', extras=None):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    if extras == True:
                        state['dev_grad'] = p.grad.data.clone()
                    else:
                        state['dev_grad'] += p.grad.data.clone()

    def multi_dev_grad_finalize(self, utility='ave', extras=None):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    state['dev_grad'].div_(extras)

    def save_train_grad_id(self, i):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if 'train_grad' not in state:
                    state['train_grad'] = [None for _ in range(len(self.args.lang_pairs))]
                if state['train_grad'][i] is None:
                    state['train_grad'][i] = p.grad.data.clone()
                else:
                    # state['train_grad'][i] = p.grad.data.clone()
                    state['train_grad'][i] = self.args.a1 * p.grad.data + self.args.a0 * state['train_grad'][i]

    def get_grad_sim_id(self, i):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                cosine_prod += (state['train_grad'][i] * p.grad.data).sum().item()
                cosine_norm += p.grad.data.norm(2) ** 2
                dev_cosine_norm += state['train_grad'][i].norm(2) ** 2
        if self.args.grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm * dev_cosine_norm) ** 0.5 + 1e-10)
            return cosine_sim.item(), cosine_norm, dev_cosine_norm
        elif self.args.grad_sim == "dot_prod":
            cosine_sim = cosine_prod
            return cosine_sim, cosine_norm, dev_cosine_norm

    def save_train_grad(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone() - state['dev_grad']

    def save_train_grad_t0(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

                # print(p.grad.data)
    def save_candidate_grad(self, keys):
        if keys == "original":
            state_key = "ori"
        elif keys == "noisy":
            state_key = "nos"
        elif keys == 0:
            state_key = "nos0"
        elif keys == 1:
            state_key = "nos1"
        elif keys == 2:
            state_key = "nos2"
        elif keys == 3:
            state_key = "nos3"
        elif keys == 4:
            state_key = "nos4"
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                if state_key in state.keys():
                    state[state_key] = state[state_key] + p.grad.data.clone()
                else:
                    state[state_key] = p.grad.data.clone()

    def just_save_grad(self, keys):
        if keys == "original":
            state_key = "ori"
        elif keys == "noisy":
            state_key = "nos"
        elif keys == 0:
            state_key = "nos0"
        elif keys == 1:
            state_key = "nos1"
        elif keys == 2:
            state_key = "nos2"
        elif keys == 3:
            state_key = "nos3"
        elif keys == 4:
            state_key = "nos4"
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                state[state_key] = p.grad.data.clone()
    # def weighted_sum(self, ori_nos_weight, nos_weight):
    #     for group in self.optimizer.param_groups:
    #         for p in group["params"]:
    #             state = self.optimizer.state[p]
    #             weights_grad = ori_nos_weight[0] * state["ori"] + ori_nos_weight[1] * (
    #                         nos_weight[0] * state["nos0"] + nos_weight[1] * state["nos1"] + nos_weight[2] * state[
    #                     "nos2"] + nos_weight[3] * state["nos3"] + nos_weight[4] * state["nos4"])
    #             p.grad.data = weights_grad.clone()

    def get_train_gradient(self):
        import copy
        state = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state.append(copy.deepcopy(p.grad.data))
        return state

    def save_dev_grad(self):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

    def clone_param_theta_t_one(self):
        """Save a copy of the params"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                state['param_copy'] = p.data.clone()

    def clone_param_theta_t(self):
        """Save a copy of the params"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                state['param_copy_theta_t'] = p.data.clone()

    def add_grad(self, eta):
        """add grad to current param"""
        # print(eta)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # print(p.grad)
                if p.grad is None: continue
                state = self.optimizer.state[p]
                # print(state['dev_grad'])
                # print(state['dev_grad'] * eta)
                p.data += state['dev_grad'] * eta

    def remove_grad(self, eta):
        """add grad to current param"""
        # print(eta)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                p.data += state['dev_grad'] * -eta

    def switch_param(self, clear_cache=False):
        """Swap copy and the param values"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                cur_p = p.data
                p.data = state['param_copy']
                if clear_cache:
                    state['param_copy'] = None
                else:
                    state['param_copy'] = cur_p

    def clear_cache(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                if 'ori' in state.keys():
                    del state['ori']
                if 'nos' in state.keys():
                    del state['nos']
                if 'nos0' in state.keys():
                    del state['nos0']
                if 'nos1' in state.keys():
                    del state['nos1']
                if 'nos2' in state.keys():
                    del state['nos2']
                if 'nos3' in state.keys():
                    del state['nos3']
                if 'nos4' in state.keys():
                    del state['nos4']
                if "dev_grad" in state.keys():
                    del state["dev_grad"]

    def print_state(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                print(state.keys())


    def get_grad_sim(self):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                cosine_prod += (state['dev_grad'] * p.grad.data).sum().item()
                cosine_norm += p.grad.data.norm(2) ** 2
                dev_cosine_norm += state['dev_grad'].norm(2) ** 2
        if self.args.grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm * dev_cosine_norm) ** 0.5 + 1e-10)
            return cosine_sim.item(), cosine_norm, dev_cosine_norm
        elif self.args.grad_sim == "positive_cosine":
            cosine_sim = cosine_prod / ((cosine_norm * dev_cosine_norm) ** 0.5 + 1e-10)
            positive_consine_sim = torch.add(cosine_sim, 1.0)
            return positive_consine_sim.item(), cosine_norm, dev_cosine_norm
        elif self.args.grad_sim == "dot_prod":
            cosine_sim = cosine_prod
            return cosine_sim, cosine_norm, dev_cosine_norm

    def _adam_delta(self, model, grads):
        deltas = {}
        for group in self.optimizer.param_groups:
            for param in group['params']:
                grad = grads[param]
                state = self.optimizer.state[param]
                p_data_fp32 = param.data.float()
                amsgrad = group['amsgrad']
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                step = state['step'] + 1

                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * param.data

                exp_avg = exp_avg * beta1 + (1. - beta1) * grad
                exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
                denom = exp_avg_sq.sqrt() + group['eps']

                bias_correction1 = 1. - beta1 ** step
                bias_correction2 = 1. - beta2 ** step
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                deltas[param] = -step_size * exp_avg / denom

        param_to_name = {param: name for name, param in model.named_parameters()}

        return {param_to_name[param]: delta for param, delta in deltas.items()}

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss, retain_graph=False):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward(retain_graph=retain_graph)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            parameters = [torch.any(p.isnan()) for p in self.params if p.grad is not None]
            if True in parameters:
                print("!!!!!!!!!!!!!!")
                print("self.params contains nan value")
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return torch.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    def average_params(self):
        pass
