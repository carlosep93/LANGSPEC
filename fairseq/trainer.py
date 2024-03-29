# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network across multiple GPUs.
"""

from collections import OrderedDict
from itertools import chain

import torch

from fairseq import distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion.cuda()
        if args.fp16:
            self._model = model.half().cuda()
        else:
            self._model = model.cuda()

        self._dummy_batch = dummy_batch
        self._num_updates = 0
        self._dis_num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._dis_optimizer = None
        self._wrapped_model = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        if args.adv == True:
            self._dis_optimizer = optim.build_optimizer(self.args, self.model.discriminator.parameters())
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds


    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        if self.args.fp16:
            if torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, self.model.parameters())

        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            utils.save_state(
                filename, self.args, self.get_model(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )


    def load_mix_checkpoints(self, enc_filename, dec_filename, enckey,deckey,newkey,reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None,freeze=False):

        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_mix_model_state(self.get_model(),enc_filename, dec_filename, enckey, deckey, newkey)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        if freeze:
            module =  self.model.decoder
            module.train(False)
            for p in module.parameters():
                p.requires_grad = False
        return extra_state

    def load_checkpoint_for_unsup_training(self,enc_filename,dec_filename, pivot_enc_filename, pivot_dec_filename, keys,pivot_keys,new_keys, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None, finetune=False,path=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_partial_unsup_model_state(enc_filename,
                                                dec_filename,
                                                pivot_enc_filename,
                                                pivot_dec_filename,
                                                self.get_model(),
                                                keys,
                                                pivot_keys, 
                                                new_keys)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        # Freeze parameters from the pivot model
        for key in self.model.models.keys():
            enc_module =  self.model.models[key].pivot_encoder
            dec_module =  self.model.models[key].pivot_decoder
            enc_module.train(False)
            dec_module.train(False)
            for p in enc_module.parameters():
                p.requires_grad = False
            for p in dec_module.parameters():
                p.requires_grad = False
            '''
            gen = self.model.models[key].generator
            for dec in gen.decoders:
                for p in dec.parameters():
                    p.requires_grad = False
            '''
        return extra_state

    def load_partial_checkpoint(self, filename,key,newkey, reuse,reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None, finetune=False,path=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_partial_model_state(filename,self.get_model(),key,newkey,reuse,finetune)
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        #Freeze pretrained part of the model
        if reuse in ['encoder','both']:
            self.model.encoder.train(False)
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        if reuse in ['decoder','both']:
            self.model.decoder.train(False)
            for p in self.model.decoder.parameters():
                p.requires_grad = False


        if reuse == 'both':
            m =    self.model.encoder.embed_tokens
            m.train(True)
            for p in m.parameters():
                p.requires_grad = True


        return extra_state


    def load_partial_audio_checkpoint(self,
                                    enc_files,
                                    dec_files,
                                    enckeys,
                                    deckeys,
                                    newkeys,
                                    reuse,
                                    reset_optimizer=False, 
                                    reset_lr_scheduler=False, 
                                    optimizer_overrides=None, 
                                    finetune=False,path=None):

        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_partial_audio_model_state(enc_files,
                                                 dec_files,
                                                 self.get_model(),
                                                 enckeys,
                                                 deckeys,
                                                 newkeys,
                                                 reuse)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()
        '''
        #Freeze pretrained part of the model
        if reuse in ['encoder','both']:
            self.model.encoder.train(False)
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            #Train  adapter network even when the encoder is frozen
            if self.model.encoder.adapt:
                self.model.encoder.adapter.train(True)
                for p in self.model.encoder.adapter.parameters():
                    p.requires_grad = True
            
        if reuse in ['decoder','both']:
            self.model.decoder.train(False)
            for p in self.model.decoder.parameters():
                p.requires_grad = False
        '''
        return extra_state





    def load_nli_encoder(self,task,model):
        ref_checkpoint,hyp_checkpoint = task.encoder_paths()
        ref_key,hyp_key = task.encoder_keys()
        utils.get_nli_encoder(model,ref_key,ref_checkpoint,'ref_encoder')
        utils.get_nli_encoder(model,hyp_key,hyp_checkpoint,'hyp_encoder')
        model.ref_encoder.train(False)
        model.hyp_encoder.train(False)
        for p in self.model.ref_encoder.parameters():
            p.requires_grad = False
        for p in self.model.hyp_encoder.parameters():
            p.requires_grad = False




    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state


    def discriminator_step(self,samples,dummy_batch=False):
        seed = self.args.seed + self.get_dis_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.model.eval()
        self.model.discriminator.train()

        self.zero_grad()

        for i,sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            loss, sample_size = self.task.discriminator_train_step(
                    sample, self.model, self.optimizer,ignore_grad)

            try:
                # normalize grads by sample size
                self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

                # clip grads
                grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)

                # take an optimization step
                self.optimizer.step()
                self._num_updates += 1

                # update learning rate
                self.lr_scheduler.step_update(self._num_updates)

            except OverflowError as e:
                print('| WARNING: overflow detected, ' + str(e))
                self.zero_grad()


    def train_step(self, samples, dummy_batch=False):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            try:
                if self.args.distributed_world_size > 1:
                    # Whenever *samples* contains more than one mini-batch, we
                    # want to accumulate gradients locally and only call
                    # all-reduce in the last backwards pass. Currently the
                    # *need_reduction* flag is only supported by
                    # LegacyDistributedDataParallel.
                    if i < len(samples) - 1:
                        self.model.accumulate_grads = True
                    else:
                        self.model.accumulate_grads = False

                # forward and backward
                loss, sample_size, logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer,
                    ignore_grad
                )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_sizes, ooms = zip(*distributed_utils.all_gather_list(
                [logging_outputs, sample_sizes, ooms],
            ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.criterion
        )
        sample_size = self.task.grad_denom(sample_sizes, self.criterion)

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))

        try:
            # normalize grads by sample size
            self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)

            # take an optimization step
            self.optimizer.step()
            self._num_updates += 1

            # update learning rate
            self.lr_scheduler.step_update(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['oom'].update(ooms)
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.criterion
        )
        sample_size = self.task.grad_denom(
            sample_size, self.criterion
        )

        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def get_dis_num_updates(self):
        return self._dis_num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)
