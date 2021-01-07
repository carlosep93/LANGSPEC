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
from fairseq.sequence_encoder import SequenceEncoder
import json
import numpy as np
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

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    
    key = None
    if args.task != 'translation':
        key = args.source_lang + '-' + args.target_lang

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),pair=key)

    for model in models:
        model.keys = [key]

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()

    encoder = SequenceEncoder(models, task.target_dictionary)


    if use_cuda:
        encoder.cuda()

    pad = options.eval_bool(args.pad)

    # Generate and compute BLEU score
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        encodings = encoder.encode_batched_itr(t, cuda=use_cuda, timer=gen_timer,pad=pad)
        data = {}
        i = 0
        for id,src,ref,hypos in encodings:
            if i >= args.n_points:
                break
            data[str(id.cpu().data.numpy())] = {
                'src':src.cpu().data.numpy().tolist(),
                'ref':ref.cpu().data.numpy().tolist(),
                'encoding':hypos[0]['encoding'].cpu().data.numpy().tolist()
            }
            i += 1
    with open(args.output_file,'w') as f:
        json.dump(data,f)
    print('Done')


if __name__ == '__main__':
    parser = options.get_generation_parser()
    options.add_encode_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)
