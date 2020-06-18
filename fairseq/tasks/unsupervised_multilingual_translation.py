# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os

import torch

from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset, RoundRobinZipDatasets,
)

from fairseq.models.fairseq_model import FairseqMultiUnsupModel
from fairseq.models.fairseq_model import  FairseqInterlinguaModel

from . import FairseqTask, register_task


@register_task('unsupervised_multilingual_translation')
class UnsupervisedMultilingualTranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--pivot-lang-pairs', default=None, metavar='PPAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')

    def __init__(self, args, dicts, pivot_dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.pivot_dicts = pivot_dicts
        self.langs = list(dicts.keys())
        self.pivot_langs = list(pivot_dicts.keys())
        self.training = training

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        if args.source_lang is not None or args.target_lang is not None:
            if args.lang_pairs is not None:
                raise ValueError(
                    '--source-lang/--target-lang implies generation, which is '
                    'incompatible with --lang-pairs'
                )
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.lang_pairs = args.lang_pairs.split(',')
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        langs = [lp.split('-')[0] for lp in args.lang_pairs]
        pivot_langs = [lp.split('-')[1] for lp in args.lang_pairs]

        # load dictionaries
        dicts = OrderedDict()
        for lang in langs :
            dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs[0]].pad()
                assert dicts[lang].eos() == dicts[langs[0]].eos()
                assert dicts[lang].unk() == dicts[langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

        #load pivot languages dictionaries
        pivot_dicts = OrderedDict()
        for lang in pivot_langs:
            pivot_dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert pivot_dicts[lang].pad() == pivot_dicts[pivot_langs[0]].pad()
                assert pivot_dicts[lang].eos() == pivot_dicts[pivot_langs[0]].eos()
                assert pivot_dicts[lang].unk() == pivot_dicts[pivot_langs[0]].unk()

            print('| [{}] dictionary: {} types'.format(lang, len(pivot_dicts[lang])))
            print('| [{}] dictionary: {} types'.format(lang, len(pivot_dicts[lang])))

        return cls(args, dicts, pivot_dicts, training)

    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None

        def sort_lang_pair(lang_pair):
            return '-'.join(sorted(lang_pair.split('-')))

        src_datasets = {}
        for lang_pair in self.args.lang_pairs:
            src, pivot = lang_pair.split('-')
            if split_exists(split, src, pivot, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, pivot))
            elif split_exists(split, pivot, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, pivot, src))
            else:
                continue
            print(lang_pair,src, list(self.dicts.keys()))
            src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
            print('| {} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))

        if len(src_datasets) == 0:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def monolingual_dataset(lang_pair):
            '''
            src, pivot = lang_pair.split('-')
            dataset = src_datasets[lang_pair]
            sizes = dataset.sizes

            return MonolingualDataset(
            dataset, sizes, self.dicts[src], self.dicts[src],add_eos_for_other_targets=False,shuffle=True)
            '''
            src, pivot = lang_pair.split('-')
            dataset = src_datasets[lang_pair]
            sizes = dataset.sizes
            return LanguagePairDataset(
                dataset, dataset.sizes, self.dicts[src],
                dataset, dataset.sizes, self.dicts[src],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, monolingual_dataset(lang_pair))
                for lang_pair in self.args.lang_pairs
            ]),
            eval_key=None if self.training else self.args.lang_pairs[0],
        )

    def build_dataset(self, tokens, lengths, src_dict):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, LanguagePairDataset(tokens, lengths, src_dict))
            ]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiUnsupModel):
            raise ValueError('MultilingualUnsupervisedTranslationTask requires a FairseqMultiUnsupModel architecture')
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        for lang_pair in self.args.lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in self.args.lang_pairs
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        #This task performs auto-encoding through iterative backtranslation over the trained modules.
        # therefore source == target
        return self.dicts[self.args.source_lang]
