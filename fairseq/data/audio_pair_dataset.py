# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, src_audio=False, audio_features=256
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_matrices(key, left_pad):
        values = [s[key] for s in samples]
        max_length = max(v.size(0) for v in values)
        res = values[0].new(len(values), max_length, values[0].size(1)).fill_(0)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)


        for i, v in enumerate(values):
            copy_tensor(v, res[i, max_length - v.size(0):, :] if left_pad else res[i, :v.size(0), :])
        return res

    id = torch.LongTensor([s['id'] for s in samples])
    if src_audio:
        src_tokens = merge_matrices('source', left_pad=left_pad_source)
        src_lengths = torch.LongTensor([s['source'].size(0) for s in samples])
    else:
        src_tokens = merge('source', left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    batch = {
                'id': id,
                'nsentences': len(samples),
                'ntokens': ntokens,
                'net_input': {
                         'src_tokens': src_tokens,
                         'src_lengths': src_lengths,
                },
        'target': target.long(),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.long()
    else:
        batch['net_input']['prev_output_tokens'] = None
    return batch


class AudioPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
        remove_eos_from_source (bool, optional): if set, removes eos from end of
            source if it's present. Default: ``False``
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent. Default: ``False``
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        src_audio=False,audio_features=256, remove_eos_from_target=False,
    ):
        if tgt_dict is not None and not src_audio:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes) if not src_audio else np.array(src_sizes[0::2])
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.remove_eos_from_target = remove_eos_from_target
        self.src_audio = src_audio
        self.audio_features = audio_features

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        if self.remove_eos_from_target:
            tgt_item = tgt_item[:-1]

        if self.remove_eos_from_source and not self.src_audio:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, src_audio=self.src_audio, audio_features=self.audio_features
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len) if self.src_dict is not None else None,
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )


class LanguagePairTokenDataset(AudioPairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        lang_id=-1,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        src_audio=False, id_map=None,
    ):
        super().__init__(src, src_sizes, src_dict,
        tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
        left_pad_source=left_pad_source, left_pad_target=left_pad_target,
        max_source_positions=max_source_positions, max_target_positions=max_target_positions,
        shuffle=shuffle, input_feeding=input_feeding,
        remove_eos_from_source=remove_eos_from_source,
        append_eos_to_target=append_eos_to_target,
        src_audio=src_audio)
        self.lang_id = lang_id
        self.id_map = id_map

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the targvalues[0].size(1)et sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        for sample in samples:
            if not self.src_audio:
                sample['net_input']['src_tokens'] = self.id_map[sample['net_input']['src_tokens']]
            sample['target'] = self.id_map[sample['target']]
        collated = collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, src_audio=self.src_audio
        )

        bz = collated['net_input']['src_tokens'].size(0)
        lang_id = collated['target'].new_full((bz,), self.lang_id).long()
        collated['net_input']['src_tokens'] = list((collated['net_input']['src_tokens'], lang_id))
        return collated
