# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

# TODO: need to add the data score to the loss of each noise
# TODO: added: 1) weights 2) change the original sentence level processing to batch level
def max_margin_loss(model, pos_ori_lprobs, pos_ori_target, n_list, ori_sample, sample_size, epsilon,
                    ignore_index=None, nos_prob=None, sub=None, key=None, weight_list=None):
    '''
    ori_lprobs: the log likelihood of P(si|ti) with shape [N, M, V]
    ori_target: the log likelihood of P(si|ti) with shape [N, M]
    ori_lprobs_other：the log likelihood of P(si'|si) with shape [N, M, V]
    ori_target_other：the log likelihood of P(si'|si) with shape [N, M]
    nll_matrix: p(ti|si)
    nll_other_matrix: p(si'|si)
    '''

    import copy
    a, b = pos_ori_target.shape
    target = pos_ori_target.view(a, b, 1)
    nll_matrix = torch.gather(-pos_ori_lprobs, 2, target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_matrix.masked_fill_(pad_mask, 0.)
    a, b, _ = nll_matrix.shape
    final_new_nll_other_matrix = []
    for i in range(n_list[0]):
        tmp_sample = copy.deepcopy(ori_sample)
        tmp_sample['net_input']['prev_output_tokens'] = ori_sample['net_input']['prev_output_tokens_neg_padded'][i]
        # tmp_sample['target'] = ori_sample['neg_target'][i]
        neg_sample = copy.deepcopy(tmp_sample)
        del neg_sample['net_input']['prev_output_tokens_neg_padded']
        neg_net_output = model(**neg_sample['net_input'])
        neg_ori_lprobs = model.get_normalized_probs(neg_net_output, log_probs=True)
        neg_ori_target = model.get_targets(neg_sample, neg_net_output)

        a, b = neg_ori_target.shape
        target_other = neg_ori_target.view(a, b, 1)
        nll_other_matrix = torch.gather(-neg_ori_lprobs, 2, target_other)

        if ignore_index is not None:
            pad_mask_other = target_other.eq(ignore_index)
            if pad_mask_other.any():
                nll_other_matrix.masked_fill_(pad_mask_other, 0.)

        a, b, _ = nll_other_matrix.shape
        new_nll_other_matrix = nll_other_matrix.view(a, b)
        final_new_nll_other_matrix.append(new_nll_other_matrix)
    final_loss = 0
    # print(len(weight_list))
    ids = tmp_sample["id"].cpu().detach().numpy()
    tmp_weight_list = []
    # tmp_weight_list = [[] for i in range(n_list[0])]
    for cand_idx in range(n_list[0]):
        tmp_weight_list.append(weight_list[cand_idx][ids])
    tmp_weight_tensor = torch.FloatTensor(tmp_weight_list)
    # print(tmp_weight_tensor.shape)
    # print(tmp_weight_tensor[0][0] + tmp_weight_tensor[1][0] + tmp_weight_tensor[2][0] + tmp_weight_tensor[3][0] + tmp_weight_tensor[4][0])
    for candi in range(n_list[0]):
        candi_weight = tmp_weight_tensor[candi].unsqueeze(1).cuda()
        weighted_loss = candi_weight * final_new_nll_other_matrix[candi]
        # print(final_new_nll_other_matrix[candi].shape)
        local_loss = weighted_loss.sum()
        final_loss = final_loss + local_loss
    return final_loss


def mml_label_smoothed_nll_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample,
                                sample_size, epsilon, ori_weight, ignore_index=None, reduce=True,
                                ori_prob=None, nos_prob=None, sub=None, keys=None, weight_list=None):
    # print("ori_weight: {}".format(ori_weight))
    pos_target = pos_ori_target.view(-1, 1)
    pos_lprobs = pos_ori_lprobs.view(-1, pos_ori_lprobs.size(-1))

    if pos_target.dim() == pos_lprobs.dim() - 1:
        pos_target = pos_target.unsqueeze(-1)
    nll_loss = -pos_lprobs.gather(dim=-1, index=pos_target)
    smooth_loss = -pos_lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = pos_target.eq(ignore_index)

        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / pos_lprobs.size(-1)
    ce_loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    if ori_prob is None:
        if keys == "original":
            # print("Just returning the CE loss")
            # print("==========================")
            return ce_loss, nll_loss
        elif keys == "noisy":  # nos_prob=None, sub=None
            mml = max_margin_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample, sample_size, epsilon,
                                  ignore_index, key=keys, weight_list=weight_list)
            # print("Returning the noisy loss for all candidates")
            # print("==========================")
            return mml, nll_loss
        elif sub is not None:  # nos_prob=None
            mml = max_margin_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample, sample_size, epsilon,
                                  ignore_index, sub=sub, weight_list=weight_list)
            # print("Returning the loss for a single candidate")
            # print("==========================")
            return mml, nll_loss
        else:
            mml = max_margin_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample, sample_size, epsilon,
                                  ignore_index, weight_list=weight_list)
            loss = ori_weight * ce_loss + (1 - ori_weight) * mml
            return loss, nll_loss
    else:  # sub=None
        mml = max_margin_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample, sample_size, epsilon,
                              ignore_index, nos_prob=nos_prob, weight_list=weight_list)
        loss = ori_prob[0] * ce_loss + ori_prob[1] * mml
        return loss, nll_loss

@register_criterion('tacl_cross_entropy')
class TACLCrossEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.ori_weight = args.ori_weight
        self.weight_path = args.weight_path
        self.weight_path_list = args.weight_path.split(":")
        self.weight_list = [[] for i in range(len(self.weight_path_list))]
        for weight_idx in range(len(self.weight_path_list)):
            weight_cont = open(self.weight_path_list[weight_idx]).readlines()
            for cont in weight_cont:
                self.weight_list[weight_idx].append(float(cont.replace("\n", "")))
        self.weight_list = numpy.asarray(self.weight_list)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ori-weight', default=0.5, type=float, metavar='D',
                            help='weight for the original data')
        parser.add_argument('--weight-path', type=str, default=None,
                            help='bohb hyper-parameter tuning')
        # fmt: on

    def forward(self, model, sample, reduce=True, ori_prob=None, nos_prob=None, update_theta=False, sub=None, keys=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if 'valid' in sample.keys():
            del sample['net_input']['prev_output_tokens_neg_padded']
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample)
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        else:
            import copy
            pos_sample = copy.deepcopy(sample)
            tmp_sample = copy.deepcopy(sample)
            del pos_sample['net_input']['prev_output_tokens_neg_padded']
            pos_net_output = model(**pos_sample['net_input'])
            n_list = pos_sample['neg_length']
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            loss, nll_loss = self.mml_compute_loss(model, n_list, pos_net_output, tmp_sample, pos_sample, sample_size,
                                                   reduce=reduce, ori_prob=ori_prob, nos_prob=nos_prob,
                                                   sub=sub, keys=keys, weight_list=self.weight_list)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def mml_compute_loss(self, model, n_list, pos_net_output, tmp_sample, pos_sample, sample_size, reduce=True,
                         ori_prob=None, nos_prob=None, sub=None, keys=None, weight_list=None):
        pos_ori_lprobs = model.get_normalized_probs(pos_net_output, log_probs=True)
        pos_ori_target = model.get_targets(pos_sample, pos_net_output)
        loss, nll_loss = mml_label_smoothed_nll_loss(model, pos_ori_lprobs, pos_ori_target, n_list, tmp_sample,
                                                     sample_size, self.eps, self.ori_weight,
                                                     ignore_index=self.padding_idx, reduce=reduce,
                                                     ori_prob=ori_prob, nos_prob=nos_prob, sub=sub, keys=keys,
                                                     weight_list=self.weight_list)
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
