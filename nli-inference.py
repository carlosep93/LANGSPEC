#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
import torch.nn as nn
from fairseq.sequence_scorer import SequenceScorer


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu


    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))


    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model = utils.load_nli_model_for_inference(args.path, task, model_arg_overrides=args.model_overrides)



    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions()]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    if use_cuda:
        model.cuda()

    preds = []
    labels = []
    # Generate and compute BLEU score
    model.eval()
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            with torch.no_grad():
                net_input = sample['net_input']
                if use_cuda:
                    net_input = {k:v.cuda() for k,v in net_input.items()}
                preds.append(model(**net_input)[0].tolist())
                labels.append(net_input['labels'][0].tolist())


    #Compute accuracy
    preds = [p.index(max(p)) for p in preds]
    labels = [l.index(max(l)) for l in labels]
    r = [1.0 if p == l else 0 for p,l in zip(preds,labels)]
    acc = sum(r)/len(r)
    print('Accuracy', acc)











if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
