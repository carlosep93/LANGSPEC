# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This ref code is licensed under the license found in the LICENSE file in
# the root directory of this ref tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os
import torch

from fairseq import options
from fairseq.data import (
    data_utils, Dictionary, NliDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)

from . import FairseqTask, register_task


@register_task('nli')
class NliTask(FairseqTask):
    """
    Translate from one (ref) language to another (hyp) language.

    Args:
        ref_dict (Dictionary): dictionary for the ref language
        hyp_dict (Dictionary): dictionary for the hyp language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-r', '--ref-lang', default=None, metavar='REF',
                            help='reference language')
        parser.add_argument('-hl', '--hyp-lang', default=None, metavar='HYP',
                            help='hypothesis language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--labels', action='store_true',
                            help='path to label file')
        parser.add_argument('--left-pad-ref', default='True', type=str, metavar='BOOL',
                            help='pad the reference on the left')
        parser.add_argument('--left-pad-hyp', default='False', type=str, metavar='BOOL',
                            help='pad the hypothesis on the left')
        parser.add_argument('--max-ref-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the reference sequence')
        parser.add_argument('--max-hyp-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the hypothesis sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--enc-path', type=str,
                            help='path to pretrained encoder file')
        parser.add_argument('--enc-key', type=str,
                            help='key (if any) of pretrained encoder')

    def __init__(self, args, ref_dict, hyp_dict):
        super().__init__(args)
        self.ref_dict = ref_dict
        self.hyp_dict = hyp_dict
        self.enc_path= args.enc_path
        self.enc_key = args.enc_key

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_ref = options.eval_bool(args.left_pad_ref)
        args.left_pad_hyp = options.eval_bool(args.left_pad_hyp)

        # find language pair automatically
        if args.ref_lang is None or args.hyp_lang is None:
            args.ref_lang, args.hyp_lang = data_utils.infer_language_pair(args.data[0])
        if args.ref_lang is None or args.hyp_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        ref_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.ref_lang)))
        hyp_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.hyp_lang)))
        assert ref_dict.pad() == hyp_dict.pad()
        assert ref_dict.eos() == hyp_dict.eos()
        assert ref_dict.unk() == hyp_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.ref_lang, len(ref_dict)))
        print('| [{}] dictionary: {} types'.format(args.hyp_lang, len(hyp_dict)))

        return cls(args, ref_dict, hyp_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, ref, hyp, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, ref, hyp, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        ref_datasets = []
        hyp_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):

            # infer langcode
            ref, hyp = self.args.ref_lang, self.args.hyp_lang
            ref_datasets.append(indexed_dataset(data_path + '/' +  '.'.join(['ref',split,ref]), self.ref_dict))
            hyp_datasets.append(indexed_dataset(data_path + '/' +  '.'.join(['hyp',split,hyp]), self.hyp_dict))

            if not combine:
                break

        assert len(ref_datasets) == len(hyp_datasets)

        if len(ref_datasets) == 1:
            ref_dataset, hyp_dataset = ref_datasets[0], hyp_datasets[0]
        else:
            sample_ratios = [1] * len(ref_datasets)
            sample_ratios[0] = self.args.upsample_primary
            ref_dataset = ConcatDataset(ref_datasets, sample_ratios)
            hyp_dataset = ConcatDataset(hyp_datasets, sample_ratios)

        labels = '.'.join([data_paths[0] + '/' + 'lab', split, ref])
        self.datasets[split] = NliDataset(
            ref_dataset, ref_dataset.sizes, self.ref_dict,
            hyp_dataset, hyp_dataset.sizes, self.hyp_dict,
            label_path=labels,
            max_reference_positions=self.args.max_ref_positions,
            max_hypothesis_positions=self.args.max_hyp_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_ref_positions, self.args.max_hyp_positions)

    def encoder_path(self):
        return self.enc_path

    def encoder_key(self):
        return self.enc_key

    @property
    def ref_dictionary(self):
        """Return the ref :class:`~fairseq.data.Dictionary`."""
        return self.ref_dict

    @property
    def hyp_dictionary(self):
        """Return the hyp :class:`~fairseq.data.Dictionary`."""
        return self.hyp_dict

    @property
    def target_dictionary(self):
        """Return the hyp :class:`~fairseq.data.Dictionary`."""
        return self.hyp_dict



    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        agg_loss += loss.detach().item()
        # TODO make summing of the sample sizes configurable
        agg_sample_size += sample_size
        agg_logging_output = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            loss, sample_size, logging_output = criterion(model, sample)
            agg_loss += loss.data.item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output = logging_output
        return agg_loss, agg_sample_size, agg_logging_output
