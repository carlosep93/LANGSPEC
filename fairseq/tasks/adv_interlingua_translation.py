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

from torch.nn import functional as F
from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset, ParallelRoundRobinZipDatasets,
)
from fairseq.models.fairseq_model import FairseqAdvInterlinguaModel

from . import FairseqTask, register_task




@register_task('adv_interlingua_translation')
class AdvInterlinguaTranslationTask(FairseqTask):
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
        parser.add_argument('--d_kernel_size', default=2, type=int, metavar='N',
                            help='kernel size of the discriminator')
        parser.add_argument('--d_stride', default=1, type=int, metavar='N',
                            help='discriminator conv stride')
        parser.add_argument('--dis_clip', default=1, type=int, metavar='N',
                            help='discriminator clipping factor')

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
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
        if not isinstance(model, FairseqAdvInterlinguaModel):
            raise ValueError('AdvInterlinguaTranslationTask requires a FairseqAdvInterlinguaModel architecture')
        return model

    def pad_sample(self,sample):
        max_length = max([sample[key]['net_input']['src_lengths'].data.tolist()[0] for key in sample.keys()])
        for key in sample.keys():
            npad = max_length - sample[key]['net_input']['src_lengths'].data.tolist()[0]
            if npad != 0:
                sample[key]['net_input']['src_tokens'] = torch.nn.functional.pad(sample[key]['net_input']['src_tokens'],(0,npad),'constant',0)
                sample[key]['net_input']['src_lengths'] = torch.IntTensor([max_length]*len(sample[key]['net_input']['src_lengths']))
                sample[key]['ntokens'] = max_length
        return sample


    def clip_parameters(self,model, clip):
        """
        Clip model weights.
        """
        if clip > 0:
            for x in model.parameters():
                x.data.clamp_(-clip, clip)



    def discriminator_train_step(self,sample,model,optimizer,ignore_grad=False):
        discriminator = model.discriminator
        model.eval()
        discriminator.train()
        sample = self.pad_sample(sample)
        encoded = []
        for lp in self.args.lang_pairs:
                ni = sample[lp]['net_input']
                e = model.models[lp].encoder(ni['src_tokens'],ni['src_lengths'])['encoder_out']
                encoded.append(e)
            
       # discriminator
        #dis_inputs = [x.view(-1, x.size(-1)) for x in encoded]
        dis_inputs = [x.permute(1,0,2) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        dis_inputs = torch.cat(dis_inputs,0)
        predictions = discriminator(dis_inputs).squeeze(1)

        # loss
        dis_target = torch.cat([torch.zeros(sz) for i, sz in enumerate(ntokens)])
        y = torch.Tensor([0.0 if i.data == 0 else 1.0 for i in dis_target])
        y = y.contiguous().float().cuda()

        loss = model.discriminator_criterion(predictions, y)
        #self.stats['dis_costs'].append(loss.item())

        # optimizer
        discriminator.zero_grad()
        loss.backward()
        optimizer.step()
        self.clip_parameters(discriminator, 0.1)
        return loss, ntokens[0]

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        model.discriminator.eval()
        sample = self.pad_sample(sample)
        losses = []
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))
        ae_langs = [l+'-'+l for l in langs]

        for lang_pair in self.args.lang_pairs + ae_langs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair],model.discriminator)
            if ignore_grad:
                loss *= 0
            losses.append(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output


        #discriminator loss term
        encoded = []
        for lp in self.args.lang_pairs:
                ni = sample[lp]['net_input']
                e = model.models[lp].encoder(ni['src_tokens'],ni['src_lengths'])['encoder_out']
                encoded.append(e)

        dis_inputs = [x.permute(1,0,2) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        dis_inputs = torch.cat(dis_inputs,0)
        predictions = model.discriminator(dis_inputs).squeeze(1)

        # loss
        dis_target = torch.cat([torch.zeros(sz) for i, sz in enumerate(ntokens)])
        y = torch.Tensor([0.0 if i.data == 0 else 1.0 for i in dis_target])
        y = y.contiguous().float().cuda()

        loss = model.discriminator_criterion(predictions, y)
        losses.append(loss)
        loss = loss.data.item()
        agg_loss += loss

        agg_logging_output['adv'] = {'loss':loss, 'nll_loss':loss, 'ntokens':sum(ntokens),'nsentences':len(ntokens),'sample_size':sum(ntokens)}

        optimizer.backward(sum(losses))
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        sample = self.pad_sample(sample)
        langs = [l.split('-') for l in self.args.lang_pairs]
        langs = set(reduce(lambda x, y:x+y,langs))
        ae_langs = [l+'-'+l for l in langs]
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.args.lang_pairs +  ae_langs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair],model.discriminator)
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

            encoded = []
            for lp in self.args.lang_pairs:
                ni = sample[lp]['net_input']
                e = model.models[lp].encoder(ni['src_tokens'],ni['src_lengths'])['encoder_out']
                encoded.append(e)

            dis_inputs = [x.permute(1,0,2) for x in encoded]
            ntokens = [dis_input.size(0) for dis_input in dis_inputs]
            dis_inputs = torch.cat(dis_inputs,0)
            predictions = model.discriminator(dis_inputs).squeeze(1)

            # loss
            dis_target = torch.cat([torch.zeros(sz) for i, sz in enumerate(ntokens)])
            y = torch.Tensor([0.0 if i.data == 0 else 1.0 for i in dis_target])
            y = y.contiguous().float().cuda()

            loss = model.discriminator_criterion(predictions, y)
            loss = loss.data.item()
            agg_loss += loss

            agg_logging_output['adv'] = {'loss':loss, 'nll_loss':loss, 'ntokens':sum(ntokens),'nsentences':len(ntokens),'sample_size':sum(ntokens)}

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
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in self.args.lang_pairs  + ae_langs + ['adv']
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
