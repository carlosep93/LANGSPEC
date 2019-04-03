# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils
import torch
import torch.nn.functional as F
from . import FairseqCriterion, register_criterion
from torch.nn.modules.loss import L1Loss

from fairseq.criterions.comparative_distance import ComparativeDistance

@register_criterion('pool_cosine_label_smoothed_cross_entropy')
class PoolCosineLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.dist = self.compute_dist
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self,encodings,reduce=True,distance=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = self.compute_loss(encodings,reduce=reduce)
        sample_size = encodings[0].size(1)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': encodings[0].size(1),
            'nsentences': encodings[0].size(0),
            'sample_size':  encodings[0].size(1),
        }
        return loss, sample_size, logging_output


    def compute_dist(self,X,Y,e=1e-6):
        X = X.permute(1,0,2)
        Y = Y.permute(1,0,2)
        pooled_X = F.max_pool2d(X, (X.shape[1], 1)).squeeze(1)
        pooled_Y = F.max_pool2d(Y, (Y.shape[1], 1)).squeeze(1)
        dist = self.cos(pooled_X,pooled_Y)
        return  - dist.sum()+e



    def compute_loss(self, encodings, reduce=True,distance=True):
        dist  = self.dist(encodings[1],encodings[0])+1e-6 if distance else 0.0
        return - dist

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
