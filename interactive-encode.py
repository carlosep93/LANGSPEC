#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import numpy as np
import sys
import json

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_encoder import SequenceEncoder


Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset(tokens, lengths, task.source_dictionary) if 'build_dataset' in dir(task) else \
                data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    #models, model_args = utils.load_ensemble_for_inference(model_paths, task, model_arg_overrides=eval(args.model_overrides))

    if args.task != 'translation':
        print('Load Partial Model')
        key = args.source_lang + '-' + args.target_lang
        models, _ = utils.load_partial_model_for_inference(args.enc_model,
                                                       args.enc_key,
                                                       args.dec_model,
                                                       args.dec_key,
                                                       args.newkey,
                                                       args.newarch,
                                                       args.newtask,
                                                       task,
                                                       model_arg_overrides=eval(args.model_overrides),
                                                       pair=key)


        for model in models:
            model.keys = [key]

    else:
        models, _ = utils.load_ensemble_for_inference(model_paths, task, model_arg_overrides=eval(args.model_overrides))

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Initialize generator
    encoder = SequenceEncoder(models, task.target_dictionary)

    if use_cuda:
        encoder.cuda()
    
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    data = {}
    current_idx = 0
    for inputs in buffered_read(args.buffer_size):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, args, task, max_positions):
            tokens = batch.tokens
            lengths = batch.lengths

            if use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()

            encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
            encodings = encoder.encode_interactive(encoder_input,args.maxlength)

            data[str(current_idx)] = {
                'src':tokens.cpu().data.numpy().tolist(),
                'encoding':encodings['encoder_out'].cpu().data.numpy().tolist()
            }
            current_idx += 1

    print(current_idx, len(data))
    with open(args.output_file,'w') as f:
        json.dump(data,f)
    print('Done')
    
if __name__ == '__main__':
    parser = options.get_generation_add_lang_parser(interactive=True)
    options.add_encode_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)
