# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
from functools import reduce
import numpy as np

import os

import torch

from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset, ParallelRoundRobinZipDatasets,
)
from fairseq.models.fairseq_model import FairseqInterlinguaModel
from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets,ParallelRoundRobinZipDatasets

from . import FairseqTask, register_task

from fairseq.criterions.correlation_distance import CorrelationDistance
from fairseq.criterions.pool_cosine_label_smoothed_cross_entropy import  PoolCosineLabelSmoothedCrossEntropyCriterion




@register_task('interlingua_translation')
class InterlinguaTranslationTask(FairseqTask):
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
        parser.add_argument('--distance',default='False', type=str, metavar='BOOL',help='Flag for fine tune weigths')
        parser.add_argument('--lbd', default=1, type=int, metavar='N',
                            help='regulation constant for the correlation distance')
        parser.add_argument('--auto-encoding', default='True', type=str, metavar='BOOL',
                            help='use autoencoders during training')





    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.training = training
        self.auto = args.auto_encoding == 'True'
        if args.distance == 'True':
            self.dist = PoolCosineLabelSmoothedCrossEntropyCriterion(args,self)
        else:
            self.dist = None

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

        langs = list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')})

        # load dictionaries
        dicts = OrderedDict()
        for lang in langs:
            dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs[0]].pad()
                assert dicts[lang].eos() == dicts[langs[0]].eos()
                assert dicts[lang].unk() == dicts[langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))


        #lang_split = args.lang_pairs[0].split('-')
        #args.lang_pairs += [lang_split[1] + '-' + lang_split[0]]  + [lang_split[0] + '-' + lang_split[0]] + [lang_split[1] + '-' + lang_split[1]]
        return cls(args, dicts, training)

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

        src_datasets, tgt_datasets = {}, {}
        for lang_pair in set(map(sort_lang_pair, self.args.lang_pairs)):
            src, tgt = lang_pair.split('-')
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif split_exists(split, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                continue
            src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
            tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt])
            print('| {} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))

        if len(src_datasets) == 0:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            if lang_pair in src_datasets:
                src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            else:
                lang_pair = sort_lang_pair(lang_pair)
                tgt_dataset, src_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.dicts[src],
                tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )

        def autoencoder_dataset(lang,lang_pair):
            src, tgt = lang_pair.split('-')
            if lang_pair in src_datasets:
                src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            else:
                lang_pair = sort_lang_pair(lang_pair)
                tgt_dataset, src_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            if lang == src:
                return LanguagePairDataset(
                    src_dataset, src_dataset.sizes, self.dicts[src],
                    src_dataset, src_dataset.sizes, self.dicts[src],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                )
            else:
                return LanguagePairDataset(
                    tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                    tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                )

        datasets = OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.args.lang_pairs
            ])

        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))


        if self.auto:
            for l in langs:
                datasets[l+'-'+l] = autoencoder_dataset(l,self.args.lang_pairs[0])

        print('*************************')
        print(datasets.keys())
        print('*************************')

        self.datasets[split] = ParallelRoundRobinZipDatasets(
            datasets,
            eval_key=None if self.training else self.args.lang_pairs[0],
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqInterlinguaModel):
            raise ValueError('InterlinguaTranslationTask requires a FairseqInterlinguaModel architecture')
        return model

    def squareform(self, X, e=1e-6):
        n = X.shape[0]
        d = X.shape[1]
        X1 = X.unsqueeze(1).expand(n,n,d)
        X2 = X.unsqueeze(0).expand(n,n,d)
        dist = torch.pow(X1 - X2, 2).sum(2)
        return torch.sqrt(dist + e)

    def correlation_distance(self,X, Y,e=1e-6):
        X = X.permute(1,0,2)
        Y = Y.permute(1,0,2)
        X = X.contiguous()
        Y = Y.contiguous()
        X = X.view(X.shape[0],-1)
        Y = Y.view(Y.shape[0],-1)
        n = X.shape[0]
        a = self.squareform(X)
        b = self.squareform(Y)
        A = a - a.mean(dim=0).unsqueeze(0) - a.mean(dim=1).unsqueeze(-1) + a.mean()
        B = b - b.mean(dim=0).unsqueeze(0) - b.mean(dim=1).unsqueeze(-1) + b.mean()

        dcov2_xy = (A * B).sum()/float(n * n) + e
        dcov2_xx = (A * A).sum()/float(n * n) + e
        dcov2_yy = (B * B).sum()/float(n * n) + e
        dcor = torch.sqrt(dcov2_xy)/ torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy)) + e
        return dcor


    def pad_sample(self,sample):
        max_length = max([sample[key]['net_input']['src_lengths'].data.tolist()[0] for key in sample.keys()])
        for key in sample.keys():
            npad = max_length - sample[key]['net_input']['src_lengths'].data.tolist()[0]
            if npad != 0:
                sample[key]['net_input']['src_tokens'] = torch.nn.functional.pad(sample[key]['net_input']['src_tokens'],(npad,0),'constant',1)
                sample[key]['net_input']['src_lengths'] = torch.IntTensor([max_length]*len(sample[key]['net_input']['src_lengths']))
                sample[key]['ntokens'] = max_length
        return sample

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        sample = self.pad_sample(sample)
        losses = []
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))
        ae_langs = [l+'-'+l for l in langs]

        langs = self.args.lang_pairs + ae_langs if self.auto else self.args.lang_pairs

        for lang_pair in langs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
            if ignore_grad:
                loss *= 0
            losses.append(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output


        #get encodings from translation models
        if self.dist:

            encodings = []
            for lp in self.args.lang_pairs:
                ni = sample[lp]['net_input']
                e = model.models[lp].encoder(ni['src_tokens'],ni['src_lengths'])['encoder_out']
                encodings.append(e)

            nd = torch.cuda.device_count()
            d0 = torch.device("cuda:" + str(nd-1)) if nd > 1 else torch.device("cpu:0")
            loss, sample_size, logging_output  = self.dist(encodings)

            if ignore_grad:
                loss *= 0
            losses.append(loss.to(d0))
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output['int'] = logging_output


        optimizer.backward(sum(losses))
        return agg_loss, agg_sample_size, agg_logging_output
    
    def build_dataset(self, tokens, lengths, src_dict):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, LanguagePairDataset(tokens, lengths, src_dict))
            ]),
            eval_key=lang_pair,
        )    


    def valid_step(self, sample, model, criterion):
        model.eval()
        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))
        ae_langs = [l+'-'+l for l in langs]
        l_pairs = self.args.lang_pairs  + ae_langs if self.auto else self.args.lang_pairs
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in l_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

            if self.dist:

                encodings = []
                for lp in self.args.lang_pairs:
                    ni = sample[lp]['net_input']
                    e = model.models[lp].encoder(ni['src_tokens'],ni['src_lengths'])['encoder_out']
                    encodings.append(e)

                nd = torch.cuda.device_count()
                d0 = torch.device("cuda:" + str(nd-1)) if nd > 1 else torch.device("cpu:0")
                loss, sample_size, logging_output  = self.dist(encodings)
                agg_loss += loss.data.item()
                agg_logging_output['int'] = logging_output

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
        # aggregate logging outputs for each language pairlangs = [l.split('-') for l in self.args.lang_pairs]
        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))
        ae_langs = [l+'-'+l for l in langs]
        lp = self.args.lang_pairs  + ae_langs if self.auto else self.args.lang_pairs
        lp = lp + ['int'] if self.dist else lp
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in lp
        }
        #agg_logging_outputs['corr'] = logging_outputs['corr']

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
        return self.dicts[self.args.target_lang]
