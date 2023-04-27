import torch
import numpy as np
from . import data_utils, LanguagePairDataset
import errant
import random

def collate(
        samples, num_sources, sep_idx, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        remove_eos_from_target=False, input_feeding=True, src_ppl=None, tgt_ppl=None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    annotator = errant.load('en')

    id = torch.LongTensor([s['id'] for s in samples])
    ppl_tgt_list = [[] for _i in range(num_sources-2)]

    # if src_ppl is not None:
    #     ppl_src = torch.FloatTensor([float(src_ppl[s['id']].replace("\n", "")) for s in samples])
    #     for i in range(num_sources - 2):
    #         ppl_tgt_list[i] = torch.FloatTensor([float(tgt_ppl[i][s['id']].replace("\n", "")) for s in samples])
    # else:
    #     ppl_src = torch.FloatTensor([0.0 for s in samples])
    #     for i in range(num_sources - 2):
    #         ppl_tgt_list[i] = torch.FloatTensor([0.0 for s in samples])

    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    target = None
    n_list = []
    prev_output_tokens = None
    if samples[0].get('target', None) is not None:
        target_rows = [s['target'] for s in samples]
        target_list = [[] for _i in range(num_sources)]
        target_lengths = [[] for _i in range(num_sources)]
        prev_output_tokens_list = [[] for _i in range(num_sources)]
        for row in target_rows:
            if not remove_eos_from_target and row[-1] != sep_idx:
                row = torch.cat([row, torch.Tensor([sep_idx])])
            c = row == sep_idx
            eos_pos = torch.nonzero(c, as_tuple=False).squeeze()
            assert eos_pos.dim() == 1, "eos_pos dim not 1"
            eos_pos = eos_pos.tolist()
            last_eos = -1
            num_segments = 0
            for eos in eos_pos:
                if last_eos >= eos - 1:  # skip empty segment from consecutive eos, if any
                    continue
                if remove_eos_from_target:
                    segment_tgt = row[last_eos + 1:eos]
                else:
                    segment_tgt = row[last_eos + 1:eos + 1]

                target_lengths[num_segments].append(segment_tgt.numel())
                target_list[num_segments].append(segment_tgt)
                last_eos = eos
                num_segments += 1

            assert num_segments == num_sources, \
                "Source input segments ({}) is not the same as the number of encoders ({})" \
                    .format(num_segments, num_sources)
        # new_target_list = []
        target = data_utils.collate_tokens(target_list[0], pad_idx, eos_idx, left_pad_target, False)
        ntokens = sum(target_lengths[0])
        if input_feeding:
            prev_output_tokens_list[0] = data_utils.collate_tokens(target_list[0], pad_idx, eos_idx, left_pad_target,
                                                                   True)

        # print("Target list: {}".format(target_list[-1]))
        # a = "a"
        # print(a + 1)
        encoded_index = target_list[-1]
        # n_dict = {31964: 1, 31969: 2, 31975: 3, 31977: 4, 31976: 5}  # for kakao
        # n_dict = {928: 0, 6925: 1, 447: 2, 4369: 3, 3714: 4, 4168: 5}  # for gec-pseudodata
        n_dict = {101: 1, 111: 2, 142: 3, 174: 4, 176: 5}  # for SOTA
        # n_dict = {113: 1, 10431: 2, 1629: 3, 207: 4, 947: 5}  # for bart
        for i in range(len(encoded_index)):
            n_list.append(n_dict[int(encoded_index[i][0])])
        prev_output_tokens = prev_output_tokens_list[0]

        padded_decoder_input = [[] for _i in range(num_sources - 2)]
        pos_target = target_list[0]
        # pad the index
        for candi_ind in range(1, num_sources - 1):
            tmp_neg_target = target_list[candi_ind][:]
            for sent_id in range(len(pos_target)):
                if len(pos_target[sent_id]) == len(tmp_neg_target[sent_id]):
                    padded_decoder_input[candi_ind - 1].append(tmp_neg_target[sent_id])
                # pad and truncate the negative sample, if not equal to target
                else:
                    decoder_output_list = pos_target[sent_id].tolist()
                    decoder_input_list = tmp_neg_target[sent_id].tolist()
                    decoder_output = " ".join(str(e) for e in decoder_output_list)
                    decoder_input = " ".join(str(e) for e in decoder_input_list)
                    orig = annotator.parse(decoder_input)
                    cor = annotator.parse(decoder_output)
                    alignment = annotator.align(orig, cor)
                    edits = annotator.merge(alignment, merging="all-split")
                    # process edits
                    for e_idx in range(len(edits) - 1, -1, -1):
                        e = edits[e_idx]
                        decoder_input_start, decoder_input_end, decoder_input_str = e.o_start, e.o_end, e.o_str
                        _, _, decoder_output_str = e.c_start, e.c_end, e.c_str
                        if decoder_input_str == "" and decoder_output_str != "":
                            decoder_input_list.insert(decoder_input_end, pad_idx)
                        elif decoder_input_str != "" and decoder_output_str == "":
                            del decoder_input_list[decoder_input_end - 1]
                    padded_decoder_input[candi_ind - 1].append(torch.LongTensor(decoder_input_list))
        # process the converted negative sample for decoder input
        prev_output_tokens_neg_list = [[] for _i in range(num_sources - 2)]
        # print("Content for padded decoder input: {}".format(padded_decoder_input))
        if input_feeding:
            for i in range(num_sources - 2):
                prev_output_tokens_neg_list[i] = data_utils.collate_tokens(padded_decoder_input[i], pad_idx, eos_idx,
                                                                           left_pad_target, True)
    else:
        ntokens = sum(len(s['source']) for s in samples)


    # batch = {
    #     'id': id,
    #     'nsentences': len(samples),
    #     'ntokens': ntokens,
    #     'net_input': {
    #         'src_tokens': src_tokens,
    #         'src_lengths': src_lengths,
    #     },
    #     'target': target,
    #     'neg_length': n_list,
    #     'gpt2_ppl': {
    #         'src': ppl_src,
    #         'tgt': ppl_tgt_list,
    #     }
    # }
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'neg_length': n_list,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['prev_output_tokens_neg_padded'] = prev_output_tokens_neg_list
    return batch


class MultiTargetPairDataset(LanguagePairDataset):

    def __init__(
            self, num_sources, sep_idx, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_target=False, append_eos_to_target=False,
            src_ppl=None, tgt_ppl=None,
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt, tgt_sizes, tgt_dict,
            left_pad_source, left_pad_target,
            max_source_positions, max_target_positions,
            shuffle, input_feeding, remove_eos_from_target, append_eos_to_target,
        )
        self.remove_eos_from_target = remove_eos_from_target
        self.num_sources = num_sources
        self.sep_idx = sep_idx
        self.src_ppl = src_ppl
        self.tgt_ppl = tgt_ppl

    def collater(self, samples):
        """
        Merge a list of samples to form a mini-batch.
        Modified from Language Pair Dataset to return list of src_tokens instead
        """
        return collate(
            samples, num_sources=self.num_sources, sep_idx=self.sep_idx, pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            remove_eos_from_target=self.remove_eos_from_target, input_feeding=self.input_feeding, src_ppl=self.src_ppl,
            tgt_ppl=self.tgt_ppl,
        )

    # TODO: CHN: need to check about multiple targets
    def get_valid_sample(self, datasets, num=400, max_count=4096):
        sample_indices = random.sample(range(len(datasets)), num)
        samples, count = [], 0
        for i in sample_indices:
            samples.append(datasets[i])
            count += datasets[i]['source'].numel()
            if count >= max_count:
                break
        return self.collater(samples)
