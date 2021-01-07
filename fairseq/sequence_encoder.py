# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import utils


class SequenceEncoder(object):
    """Scores the target for a given source sentence."""

    def __init__(self, models, tgt_dict):
        self.models = models
        self.pad = tgt_dict.pad()

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def pad_sample(self, sample, max_length):
        for key in sample.keys():
            npad = max_length - sample['net_input']['src_lengths'].data.tolist()[0]
            if npad != 0:
                sample['net_input']['src_tokens'] = torch.nn.functional.pad(sample['net_input']['src_tokens'],(npad,0),'constant',1)
                sample['net_input']['src_lengths'] = torch.IntTensor([max_length]*len(sample['net_input']['src_lengths']))
                sample['ntokens'] = max_length
        return sample

    def pad_sample_interactive(self, sample, max_length):
        for key in sample.keys():
            npad = max_length - sample['src_lengths'].data.tolist()[0]
            if npad != 0:
                sample['src_tokens'] = torch.nn.functional.pad(sample['src_tokens'],(npad,0),'constant',1)
                sample['src_lengths'] = torch.IntTensor([max_length]*len(sample['src_lengths']))
        return sample

    def encode_batched_itr(self, data_itr, max_length=220 ,cuda=False, timer=None,pad=True):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            #s = self.pad_sample(s,max_length)
            if timer is not None:
                timer.start()
            encodings = self.encode(s)
            for i, id in enumerate(s['id'].data):
                # remove padding from ref
                if isinstance(s['net_input']['src_tokens'], list):
                    src = utils.strip_pad(s['net_input']['src_tokens'][0].data[i, :], self.pad)
                else:
                    src = utils.strip_pad(s['net_input']['src_tokens'].data[i, :], self.pad)

                #src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                encoding_i = encodings['encoder_out'][i]
                print(encoding_i.shape)
                '''
                if attn is not None:
                    attn_i = attn[i]
                    _, alignment = attn_i.max(dim=0)
                else:
                    attn_i = alignment = None
                '''
                hypos = [{
                    'tokens': src,
                    'encoding': encoding_i,
                    'id':id
                }]

                if timer is not None:
                    timer.stop(s['ntokens'])
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos

    def encode(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']
        print('Input shape', net_input['src_tokens'].shape)
        net_input.pop('prev_output_tokens',None)
        for model in self.models:
            with torch.no_grad():
                model.eval()
                encoder_out = model.encoder.forward(**net_input)
                encoder_out['encoder_out'] = encoder_out['encoder_out'].permute(1,0,2)
                #attn = decoder_out[1]

                return encoder_out

    def encode_interactive(self, sample,maxlength=220):
        """Score a batch of translations."""
        for model in self.models:
            with torch.no_grad():
                model.eval()
                #sample = self.pad_sample_interactive(sample,maxlength)
                encoder_out = model.encoder.forward(**sample)
                if isinstance(encoder_out['encoder_out'],tuple):
                    encoder_out['encoder_out'] = encoder_out['encoder_out'][0]
                encoder_out['encoder_out'] = encoder_out['encoder_out'].permute(1,0,2)
                #attn = decoder_out[1]

                return encoder_out

    def encode_layer_interactive(self, sample,maxlength=220, layer=-1):
        """Score a batch of translations."""
        for model in self.models:
            with torch.no_grad():
                model.eval()
                sample = self.pad_sample_interactive(sample,maxlength)
                encoder_out = model.encoder.get_layer(sample['src_tokens'], sample['src_lengths'],layer)
                encoder_out['encoder_out'] = encoder_out['encoder_out'].permute(1,0,2)
                #attn = decoder_out[1]

                return encoder_out
