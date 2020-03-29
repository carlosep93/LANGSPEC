# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch
from fairseq import utils
import numpy as np

from . import FairseqCriterion, register_criterion


@register_criterion('classification_loss')
class ClassificationLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.criterion = torch.nn.NLLLoss()
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        targets = model.get_targets(sample,net_output)
        #targets = targets.flatten()
        loss = self.criterion(net_output,targets)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        preds = [np.argmax(p) for p in net_output.tolist()]
        r = [1.0 if p == l else 0 for p,l in zip(preds,targets.tolist())]
        acc = sum(r)/len(r)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['net_input']['labels'].size(0),
            'sample_size': sample_size,
            'acc':acc
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        acc =  sum(log.get('acc', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'acc': acc
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
