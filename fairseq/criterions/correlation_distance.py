# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('correlation_distance')
class CorrelationDistance(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--lambda', default=0., type=float, metavar='D',
                            help='lambda to regulate the term')



    def forward(self, models, samples, lbd,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_input = samples[0]['net_input']
        encoding1 = models[0].encoder(net_input['src_tokens'], net_input['src_lengths'])
        net_input = samples[1]['net_input']
        encoding2 = models[1].encoder(net_input['src_tokens'], net_input['src_lengths'])
        loss = self.compute_loss(encoding1, encoding2, lbd,reduce=reduce)
        sample_size = samples[0]['target'].size(0) if self.args.sentence_avg else samples[0]['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data
        }
        return loss, sample_size, logging_output

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

    def compute_loss(self,encoding1, encoding2,lbd=1,reduce=True):
        nd =torch.cuda.device_count()
        d0 = torch.device("cuda:" + str(nd-1)) if nd > 1 else torch.device("cpu:0")
        loss =  ((1.0 - self.correlation_distance(encoding1['encoder_out'],encoding2['encoder_out'])) * lbd).to(d0)
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        losses = sum(log.get('loss', 0) for log in logging_outputs)
        nll_losses = sum(log.get('nll_loss', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        d =  {
            'loss': losses / sample_size / math.log(2),
            'nll_loss': nll_losses / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return d
