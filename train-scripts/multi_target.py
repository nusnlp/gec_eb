import sys
import random
import argparse
import itertools
from multiprocessing import Pool

_SEP = ' </s> '
_PAD = '<pad>'


def find_match(src_1, src_2, idx_1, idx_2, window, match=0, lm_1=0, lm_2=0, res_1=[], res_2=[]):
    if idx_1 >= len(src_1) and idx_2 >= len(src_2):
        assert len(res_1) == len(res_2)
        return (res_1, res_2, match)
    elif idx_1 >= len(src_1):
        left_over = len(src_2) - idx_2
        res_1 += ([_PAD] * left_over)
        res_2 += src_2[idx_2:]
        assert len(res_1) == len(res_2)
        return (res_1, res_2, match)
    elif idx_2 >= len(src_2):
        left_over = len(src_1) - idx_1
        res_1 += src_1[idx_1:]
        res_2 += ([_PAD] * left_over)
        assert len(res_1) == len(res_2)
        return (res_1, res_2, match)
    elif max(idx_1 - lm_1, idx_2 - lm_2) > window:
        return (res_1, res_2, -1)
    else:
        word_1 = src_1[idx_1]
        word_2 = src_2[idx_2]
        if word_1.lower() == word_2.lower():
            return find_match(src_1, src_2,
                              idx_1 + 1, idx_2 + 1,
                              window, match + 1,
                              idx_1, idx_2,
                              res_1 + [word_1], res_2 + [word_2])
        else:
            src_1_1, src_1_2, match_1 = find_match(src_1, src_2,
                                                   idx_1 + 1, idx_2,
                                                   window, match,
                                                   lm_1, lm_2,
                                                   res_1 + [word_1], res_2 + [_PAD])
            src_2_1, src_2_2, match_2 = find_match(src_1, src_2,
                                                   idx_1, idx_2 + 1,
                                                   window, match,
                                                   lm_1, lm_2,
                                                   res_1 + [_PAD], res_2 + [word_2])
            src_3_1, src_3_2, match_3 = find_match(src_1, src_2,
                                                   idx_1 + 1, idx_2 + 1,
                                                   window, match,
                                                   lm_1, lm_2,
                                                   res_1 + [word_1], res_2 + [word_2])
            if match_3 >= max(match_1, match_2):
                assert len(src_3_1) == len(src_3_2)
                return (src_3_1, src_3_2, match_3)
            elif match_1 >= match_2:
                assert len(src_1_1) == len(src_1_2)
                return (src_1_1, src_1_2, match_1)
            else:
                assert len(src_2_1) == len(src_2_2)
                return (src_2_1, src_2_2, match_2)


def combine(line, args):
    row = [l.strip() for l in list(line)]
    if args.max is not None or args.ratio is not None:
        max_len = min_len = None
        last_sent = None
        for idx, l in enumerate(row):
            cur_len = len(l.split())
            if (args.max is not None and cur_len > args.max):
                return None
            max_len = cur_len if max_len is None else max(max_len, cur_len)
            min_len = cur_len if min_len is None else min(min_len, cur_len)

            if args.target is not None and idx == len(row) - 1:
                continue
            if last_sent is None:
                last_sent = l
            else:
                if last_sent == l:
                    prob = random.random()
                    if prob > args.downsample:
                        return None

        if min_len == 0:
            print('[WARNING] Line {} is skipped'.format(row))
        if args.ratio is not None and (min_len == 0 or (float(max_len) / min_len) > args.ratio):
            return None
    if args.target is not None:
        target_out = row[-1]
        row = row[:-1]
    else:
        target_out = None

    if args.align:
        src_1 = row[0].split()
        src_2 = row[1].split()
        idx_1 = idx_2 = 0
        res_1, res_2, match = find_match(src_1, src_2, idx_1, idx_2, args.align_window)
        if match == -1:
            print('[WARNING] No alignment found for {} and {}. Try bigger --align_window.'.format(src_1, src_2))
            return None
        row = [" ".join(res_1), " ".join(res_2)]

    return (_SEP.join(row), target_out)


def cross_combine(chunk, args):
    sources, target = chunk
    combination = list(itertools.product(*sources))
    idxs = [range(len(lines)) for lines in sources]
    idxs = list(itertools.product(*idxs))
    chunk = list(zip(combination, idxs))
    chunk = sorted(chunk, key=lambda x: sum(x[1]))
    chunk = chunk[:args.upsample]
    chunk = [list(source) + [target] for source, idx in chunk]
    results = []
    for line in chunk:
        result = combine(line, args)
        if result is not None:
            results.append(result)
    return results


def main(args):
    file_paths = args.source
    if args.align:
        assert len(file_paths) == 2, "Can only accept 2 sources if using align"

    out_path = getattr(args, 'out', file_paths[0].split('.')[0]) + '.src'
    target_out = getattr(args, 'out', None)
    lines = []
    total_row = -1
    for f_path in file_paths:
        print('Reading {}...'.format(f_path))
        with open(f_path) as f:
            content = f.readlines()
            lines.append(content)
        if total_row == -1:
            total_row = len(content)
        else:
            assert total_row == len(content), \
                "The length of file {} ({}) is not the same as the previous source ({})" \
                    .format(f_path, len(content), total_row)

    if args.target is not None:
        print('Reading {}...'.format(args.target))
        with open(args.target) as f:
            target = f.readlines()
        assert len(target) * args.rep == total_row, "{} x {} != {}".format(len(target), args.rep, total_row)
    else:
        target = None

    print('Splitting into chunks...')
    if args.target is not None:
        chunks = [([None] * len(lines), line) for line in target]
    else:
        chunks = [([None] * len(lines), '') for i in range(total_row // args.rep)]
    for s_id, source in enumerate(lines):
        chunk_id = 0
        for i in range(0, len(source), args.rep):
            chunks[chunk_id][0][s_id] = source[i:i + args.rep]
            chunk_id += 1

    print('Combining...')
    p = Pool(args.workers)
    result = p.starmap(cross_combine, [(chunk, args) for chunk in chunks])
    sources = [r[0] for r in itertools.chain.from_iterable(result)]
    if args.target is not None:
        targets = [r[1] for r in itertools.chain.from_iterable(result)]
        assert len(sources) == len(targets), \
            "Length of sources ({}) should be the same with targets ({})" \
                .format(len(sources), len(targets))

    print('Writing {} from originally {} sentences{}.' \
          .format(len(sources), total_row, ' with alignment' if args.align else ''))
    with open(out_path, 'w') as com_out:
        com_out.write('\n'.join(sources))
    if args.target is not None:
        with open(target_out + '.trg', 'w') as tgt_out:
            tgt_out.write('\n'.join(targets))

    print('All done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='+', help='sources to combine')
    parser.add_argument('--target', type=str, default=None,
                        help='target path if want to use ratio and max length feature')
    parser.add_argument('--max', type=int, default=None, help='max length of sentence')
    parser.add_argument('--ratio', type=float, default=None, help='max length of sentence')
    parser.add_argument('--out', type=str, default=None, help='output file path')
    parser.add_argument('--align', action='store_true', help='output file path')
    parser.add_argument('--align_window', type=int, default=3, help='output file path')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsampled ratio of lines with exact same sentences')
    parser.add_argument('--upsample', type=int, default=1, help='sample per chunk (>1 means upsampling)')
    parser.add_argument('--rep', type=int, default=1, help='repetitions in the source')
    parser.add_argument('--workers', type=int, default=1, help='number of parallel process')
    args = parser.parse_args()
    main(args)
