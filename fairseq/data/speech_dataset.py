# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import struct

import numpy as np
import torch

from fairseq.SparseImageWarp import *

import librosa
import wave
import subprocess
import random

from fairseq.tokenizer import Tokenizer


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'



def wav_read(pipe):
    if pipe[-1] == '|':
        tpipe = subprocess.Popen(pipe[:-1], shell=True, stderr=DEVNULL, stdout=subprocess.PIPE)
        audio = tpipe.stdout
    else:
        tpipe = None
        audio = pipe
    try:
        wav = wave.open(audio, 'r')
    except EOFError:
        print('EOFError:', pipe)
        exit(-1)
    sfreq = wav.getframerate()
    assert wav.getsampwidth() == 2
    wav_bytes = wav.readframes(-1)
    npts = len(wav_bytes) // wav.getsampwidth()
    wav.close()
    # convert binary chunks
    wav_array = np.array(struct.unpack("%ih" % npts, wav_bytes), dtype=float) / (1 << 15)
    return wav_array, sfreq


def get_segment(wav, seg_ini, seg_end):
    nwav = None
    if float(seg_end) > float(seg_ini):
        if wav[-1] == '|':
            nwav = wav + ' sox -t wav - -t wav - trim {} ={} |'.format(seg_ini, seg_end)
        else:
            nwav = 'sox {} -t wav - trim {} ={} |'.format(wav, seg_ini, seg_end)
    return nwav




def make_dataset(kaldi_path):

    wavs = []
    with open(kaldi_path, 'rt') as wav_scp:
        key_to_wav = dict()
        for wav_line in wav_scp:
            wav_key, wav = wav_line.strip().split(' ', 1)
            #text_key, sentence = text_line.strip().split(' ', 1)
            #assert wav_key == text_key
            wavs.append([wav_key, wav])

    return wavs


def mel_spectrogram(path, window_size, window_stride, window, normalize, max_len,n_mels=80):
    y, sfr = wav_read(path)

    # window length
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 20
    highfreq = sfr/2 - 400

    # melspectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
    D = np.abs(S)
    param = librosa.feature.melspectrogram(S=D, sr=sfr, n_mels=n_mels, fmin=lowfreq, fmax=highfreq, norm=None)
    param = torch.FloatTensor(param)

    # z-score normalization
    if normalize:
        mean = param.mean()
        std = param.std()
        if std != 0:
            param.add_(-mean)
            param.div_(std)

    return param

def spec_augment(sample,W,F,T,mf,mt,p):

    v,tau = sample.shape
    center_position = v/2
    W = min(tau,W)
    random_point = np.random.randint(low=W, high=tau - W)
    #Sparse image warping (TO DO: boundary points)
    # warping distance chose.
    w = np.random.uniform(low=0, high=W)
    control_point_locations = torch.Tensor([[[center_position, random_point]]])
    control_point_destination = torch.Tensor([[[center_position, random_point + w]]])
    sample = sample.unsqueeze(0)
    sample, _ = sparse_image_warp(sample, control_point_locations, control_point_destination)
    sample = sample.squeeze(0)
    sample = sample.squeeze(-1)
    #Frequency masking
    for i in range(mf):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, v - f)
        sample[f0:f0 + f, :] = 0

    #Time masking
    for i in range(mt):
        t = np.random.uniform(low=0.0, high=min(T,p*v))
        t = int(t)
        t0 = random.randint(0, tau - t)
        sample[:, t0:t0 + t] = 0

    return sample


class SpeechDataset(torch.utils.data.Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self,
                 path,
                 n_mels=80,
                 window_size=.02,
                 window_stride=.01,
                 window_type='hamming',
                 normalize=True,
                 max_len=97,
                 W=10,
                 F=27,
                 T=10,
                 p=1.0,
                 mf=1,
                 mt=1,
                 append_eos=True,
                 reverse_order=False,
                 fix_lua_indexing=False,
                 transform=None,
                 masked=True,
                 read_data=True):
        super().__init__()

        #Speech params
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.transform = transform
        self.masked = masked
        self.n_mels = n_mels

        #SpecAugment parameters
        self.W = W
        self.F = F
        self.T = T
        self.mf = mf
        self.mt = mt
        self.p = p

        self.fix_lua_indexing = fix_lua_indexing
        self.wavs = make_dataset(path)
        self.data_file = None
        self.size = len(self.wavs)


        #STranscription params
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path)
        self.size = len(self.wavs)

        '''
        self.read_index(path)
        if read_data:
            self.read_data(path)
        '''
    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == b'TNTIDX\x00\x00'
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        #self.data_file = open(data_file_path(path), 'rb', buffering=0)
        self.data_file = open(path, 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        key, path = self.wavs[i]
        params = mel_spectrogram(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len,self.n_mels)  # pylint: disable=line-too-long
        if self.transform:
            params = self.transform(params)
        if self.masked:
            params = spec_augment(params,self.W, self.F,self.T, self.mf, self.mt,self.p)
        return key, params

    def __len__(self):
        return self.size


    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    @staticmethod
    def exists(path):
        return os.path.exists(path)



