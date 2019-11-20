# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This reference code is licensed under the license found in the LICENSE file in
# the root directory of this reference tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_reference=True, left_pad_hypothesis=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    ref_tokens = merge('reference', left_pad=left_pad_reference)
    # sort by descending reference length
    ref_lengths = torch.LongTensor([s['reference'].numel() for s in samples])
    hyp_lengths = torch.LongTensor([s['hypothesis'].numel() for s in samples])
    ref_lengths, sort_order = ref_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    ref_tokens = ref_tokens.index_select(0, sort_order)

    hypothesis = None
    if samples[0].get('hypothesis', None) is not None:
        hypothesis = merge('hypothesis', left_pad=left_pad_hypothesis)
        hypothesis = hypothesis.index_select(0, sort_order)
        hyp_lengths = hyp_lengths.index_select(0,sort_order)
        ntokens = sum(len(s['hypothesis']) for s in samples)

    else:
        ntokens = sum(len(s['reference']) for s in samples)

    if samples[0].get('label',None) is not None:
        labels = merge('label',left_pad=False)
        labels = labels.index_select(0,sort_order)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'reference': ref_tokens,
            'ref_lengths': ref_lengths,
            'hypothesis': hypothesis,
            'hyp_lengths': hyp_lengths,
            'labels': labels
        },
        'nsentences': samples[0]['reference'].size(0),
    }
    return batch

def load_labels(path,labels_dict):
    labels = []
    with open(path) as p:
        for l in p.readlines():
            l = l.replace('\n','')
            label = [0.0,0.0,0.0]
            label[labels_dict[l]] = 1.0
            labels.append(label)
    return torch.Tensor(labels)


class NliDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        ref (torch.utils.data.Dataset): reference dataset to wrap
        ref_sizes (List[int]): reference sentence lengths
        ref_dict (~fairseq.data.Dictionary): reference vocabulary
        hyp (torch.utils.data.Dataset, optional): hypothesis dataset to wrap
        hyp_sizes (List[int], optional): hypothesis sentence lengths
        hyp_dict (~fairseq.data.Dictionary, optional): hypothesis vocabulary
        labels(string): List of label for each ref-hyp pair
        left_pad_ref (bool, optional): pad reference tensors on the left side.
            Default: ``True``
        left_pad_hyp (bool, optional): pad hypothesis tensors on the left side.
            Default: ``False``
        max_ref_positions (int, optional): max number of tokens in the reference
            sentence. Default: ``1024``
        max_hyp_positions (int, optional): max number of tokens in the hypothesis
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the hypothesiss
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
        remove_eos_from_ref (bool, optional): if set, removes eos from end of
            reference if it's present. Default: ``False``
        append_eos_to_hypothesis (bool, optional): if set, appends eos to end of
            hypothesis if it's absent. Default: ``False``
    """

    def __init__(
        self, ref, ref_sizes, ref_dict,
        hyp=None, hyp_sizes=None, hyp_dict=None,
        left_pad_reference=True, left_pad_hypothesis=False,label_path=None,
        max_reference_positions=1024, max_hypothesis_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_reference=False, append_eos_to_hypothesis=False,
    ):
        if hyp_dict is not None:
            assert ref_dict.pad() == hyp_dict.pad()
            assert ref_dict.eos() == hyp_dict.eos()
            assert ref_dict.unk() == hyp_dict.unk()
        self.ref = ref
        self.hyp = hyp
        self.ref_sizes = np.array(ref_sizes)
        self.hyp_sizes = np.array(hyp_sizes) if hyp_sizes is not None else None
        self.ref_dict = ref_dict
        self.hyp_dict = hyp_dict
        self.left_pad_reference = left_pad_reference
        self.left_pad_hypothesis = left_pad_reference
        self.labels = load_labels(label_path,{'entailment':0,'contradiction':1,'neutral':2})
        self.max_reference_positions = max_reference_positions
        self.max_hypothesis_positions = max_hypothesis_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_reference = remove_eos_from_reference
        self.append_eos_to_hypothesis = append_eos_to_hypothesis

    def __getitem__(self, index):
        hyp_item = self.hyp[index] if self.hyp is not None else None
        ref_item = self.ref[index]
        label_item = self.labels[index]
        # Append EOS to end of hyp sentence if it does not have an EOS and remove
        # EOS from end of ref sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use hyp_dataset as ref_dataset and vice versa
        '''
        if self.append_eos_to_hypothesis:
            eos = self.hyp_dict.eos() if self.hyp_dict else self.ref_dict.eos()
            if self.hyp and self.hyp[index][-1] != eos:
                hyp_item = torch.cat([self.hyp[index], torch.LongTensor([eos])])

        if self.remove_eos_from_reference:
            eos = self.ref_dict.eos()
            if self.ref[index][-1] == eos:
                ref_item = self.ref[index][:-1]
        '''
        return {
            'id': index,
            'reference': ref_item,
            'hypothesis': hyp_item,
            'label': label_item
        }

    def __len__(self):
        return len(self.ref)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `ref_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the reference sentence of shape `(bsz, ref_len)`. Padding will
                    appear on the left if *left_pad_reference* is ``True``.
                  - `ref_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each reference sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the hypothesis sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    hyp_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_hypothesis* is ``True``.

                - `hypothesis` (LongTensor): a padded 2D Tensor of tokens in the
                  hypothesis sentence of shape `(bsz, hyp_len)`. Padding will appear
                  on the left if *left_pad_hypothesis* is ``True``.
        """
        return collate(
            samples, pad_idx=self.ref_dict.pad(), eos_idx=self.ref_dict.eos(),
            left_pad_reference=self.left_pad_reference, left_pad_hypothesis=self.left_pad_hypothesis,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, ref_len=128, hyp_len=128):
        """Return a dummy batch with a given number of tokens."""
        ref_len, hyp_len = utils.resolve_max_positions(
            (ref_len, hyp_len),
            max_positions,
            (self.max_reference_positions, self.max_hypothesis_positions),
        )
        bsz = max(num_tokens // max(ref_len, hyp_len), 1)
        return self.collater([
            {
                'id': i,
                'reference': self.ref_dict.dummy_sentence(ref_len),
                'ref_lengths': torch.Tensor([ref_len]),
                'hypothesis': self.hyp_dict.dummy_sentence(hyp_len) if self.hyp_dict is not None else None,
                'hyp_lengths': torch.Tensor([hyp_len]),
                'label': torch.Tensor([1.0,0.0,0.0])
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.ref_sizes[index], self.hyp_sizes[index] if self.hyp_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.ref_sizes[index], self.hyp_sizes[index] if self.hyp_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.hyp_sizes is not None:
            indices = indices[np.argsort(self.hyp_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.ref_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.ref.prefetch(indices)
        self.hyp.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.ref, 'supports_prefetch')
            and self.ref.supports_prefetch
            and hasattr(self.hyp, 'supports_prefetch')
            and self.hyp.supports_prefetch
        )
