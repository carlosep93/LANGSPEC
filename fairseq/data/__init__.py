# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary, TruncatedDictionary, AudioDictionary
from .fairseq_dataset import FairseqDataset
from .append_eos_dataset import AppendEosDataset
from .backtranslation_dataset import BacktranslationDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import IndexedDataset, IndexedCachedDataset, IndexedInMemoryDataset, IndexedRawTextDataset, AudioIndexedInMemoryDataset
from .language_pair_dataset import LanguagePairDataset
from .audio_pair_dataset import AudioPairDataset
from .monolingual_dataset import MonolingualDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets,ParallelRoundRobinZipDatasets
from .token_block_dataset import TokenBlockDataset
#from .speech_dataset import  SpeechDataset
from .speech_pair_dataset import  SpeechPairDataset
from .nli_dataset import NliDataset
from .audio_reader import get_reader

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'AppendEosDataset',
    'BacktranslationDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedInMemoryDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'MonolingualDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'TokenBlockDataset',
    'SpeechPairDataset',
    'NliDataset',
    'get_reader'
]
