# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os
import math
import torch

from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairDataset, IndexedInMemoryDataset, IndexedCachedDataset,
    IndexedRawTextDataset, RoundRobinZipDatasets, AudioDictionary, IndexedDataset,
    AudioPairDataset
)
from fairseq.models.fairseq_model import FairseqInterlinguaModel

from . import FairseqTask, register_task


@register_task('speech_interlingua_translation')
class SpeechInterlinguaTranslationTask(FairseqTask):
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
        parser.add_argument('--freeze-schedule', default=None, metavar='PAIRS',
                            help='comma-separated list of freeze or non freeze modules in training order')
        parser.add_argument('--auto-encoding', default='False', type=str, metavar='BOOL',
                            help='use autoencoders during training')
        parser.add_argument('--adapt-schedule', default='False', type=str, metavar='BOOL',
                            help='adapt freezing schedule after each epoch')
        parser.add_argument('--audio-features', default=256, type=int, metavar='N',
                            help='number of audio features by sampling')
        parser.add_argument('--no-cache-source', default=False, action='store_true')
        parser.add_argument('--audio-input', action='store_true',
                            help='load audio input dataset')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.training = training
        self.num_updates = 0
        self.auto = args.auto_encoding == 'True'
        self.adapt = args.adapt_schedule
        pass

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.freeze_schedule = args.freeze_schedule.split(',')
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        args.adapt_schedule = options.eval_bool(args.adapt_schedule)

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
        for lang_pair in args.lang_pairs:
            speech,lang = lang_pair.split('-')
            dicts[speech] = AudioDictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(speech)))
            dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs[0]].pad()
                assert dicts[lang].eos() == dicts[langs[0]].eos()
                assert dicts[lang].unk() == dicts[langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(speech, len(dicts[speech])))
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

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

        def indexed_dataset(path, dictionary, cached=True, audio=False, audio_features=256):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            #elif IndexedInMemoryDataset.exists(path):
            #       return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            elif IndexedDataset.exists(path):
                if cached:
                    return IndexedCachedDataset(path, fix_lua_indexing=True, audio=audio)
                else:
                    return IndexedDataset(path, fix_lua_indexing=True, audio=audio,audio_features=audio_features)
            return None

        def sort_lang_pair(lang_pair):
            return '-'.join(sorted(lang_pair.split('-')))

        src_datasets, tgt_datasets = {}, {}
        for lang_pair in set(self.args.lang_pairs):
            src, tgt = lang_pair.split('-')
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif split_exists(split, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                continue
            cached = not self.args.no_cache_source

            src_datasets[lang_pair] = indexed_dataset(prefix + src, \
                                                    self.dicts[src], \
                                                    cached=cached, \
                                                    audio=True, \
                                                    audio_features=self.args.audio_features)

            tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt],audio=False,cached=cached)

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
            return AudioPairDataset(
                src_dataset, src_dataset.sizes, self.dicts[src],
                tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                src_audio=self.args.audio_input
            )


        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.args.lang_pairs
            ]),
            eval_key=None if self.training else self.args.lang_pairs[0],
        )

        if self.args.audio_input:
            # saving audio features length, needed when creating the model.
            src_dataset = self.datasets[split].datasets[self.args.lang_pairs[0]].src
            self.audio_features = src_dataset.sizes[1]
            for lang_pair in self.args.lang_pairs:
                self.datasets[split].datasets[lang_pair].src_dict.audio_feautures = self.audio_features

        print('**************************')
        print(self.args.lang_pairs)
        print('**************************')

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
        print('Model type', type(model))
        if not isinstance(model, FairseqInterlinguaModel):
            raise ValueError('InterlinguaNoDistanceTranslationTask requires a FairseqInterlinguaModel architecture')
        return model


    def freeze_module(self,model,lang_pair,schedule):
        #freeze encoder if required by schedule
        if schedule[0] == 'f':
            model.models[lang_pair].encoder.eval()
            for p in model.models[lang_pair].encoder.parameters():
                p.requires_grad = False
        #freeze decoder if required by schedule
        if schedule[1] == 'f':
            model.models[lang_pair].decoder.eval()
            for p in  model.models[lang_pair].decoder.parameters():
                p.requires_grad = False
        return model

    def unfreeze_module(self,model,lang_pair,schedule):
        #unfreeze encoder if required by schedule
        if schedule[0] == 'f':
            model.models[lang_pair].encoder.train()
            for p in model.models[lang_pair].encoder.parameters():
                p.requires_grad = True
        #unfreeze decoder if required by schedule
        if schedule[1] == 'f':
            model.models[lang_pair].decoder.train()
            for p in  model.models[lang_pair].decoder.parameters():
                p.requires_grad = True
        return model

    def train_step(self, sample, model, criterion, optimizer,ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        for i,lang_pair in enumerate(self.args.lang_pairs):
            schedule = self.args.freeze_schedule[i].split('-')
            model = self.freeze_module(model,lang_pair,schedule)

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
            model = self.unfreeze_module(model,lang_pair,schedule)
        return agg_loss, agg_sample_size, agg_logging_output


    def adapt_schedule(self,logging,n):
        # find lang pairs without repetition
        lang_pairs =  {}
        for lp in self.args.lang_pairs:
            rev_lp =  '-'.join(lp.split('-')[::-1])
            print(lp,rev_lp)
            if not lp in lang_pairs and not rev_lp in lang_pairs:
                lang_pairs[lp] = (logging[lp + ':loss'] + logging[rev_lp + ':loss']) / 2.0

        # Select pair that will not be frozen the following epoch
        # criterion: Maximum loss pair that does not overlap with previously
        # chosen pairs
        aux_pairs = lang_pairs.copy()
        no_freeze_pairs = []
        for _ in range(math.ceil(n/2.0)):
            max_pair = max(list(aux_pairs.items()), key= lambda l:l[1])
            l1,l2 = max_pair[0].split('-')
            del_keys = [k for k in aux_pairs.keys() if l1 in k or l2 in k]
            for k in del_keys:
                del aux_pairs[k]
            no_freeze_pairs.append(max_pair[0])
            no_freeze_pairs.append('-'.join(max_pair[0].split('-')[::-1]))

        # Create frozen paths between the rest of language pairs
        # to ensure the flow of information between all languages
        freeze_pairs = []
        freeze_sc = []
        curr = first = list(self.dicts.keys())[0]
        end = len(self.dicts.keys()) == 2
        while not end:
            for l in self.dicts.keys():
                if '-'.join([curr,l]) not in no_freeze_pairs and '-'.join([curr,l]) not in freeze_pairs and  not curr == l:
                    freeze_pairs.append('-'.join([curr,l]))
                    freeze_sc.append('n-f')
                    freeze_pairs.append('-'.join([l,curr]))
                    freeze_sc.append('f-n')
                    curr = l
                    end = curr == first
                    break

        #Close the circle with the following language
        pairs = no_freeze_pairs
        schedule = ['n-n']*len(pairs)
        pairs += freeze_pairs
        schedule += freeze_sc

        return pairs, schedule


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
        return self.dicts[self.args.target_lang]
