# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import defaultdict, OrderedDict
import logging
import os
import re
import torch
import traceback
import numpy as np

from torch.serialization import default_restore_location


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(filename, args, model, criterion, optimizer, lr_scheduler,
               num_updates, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'last_optimizer_state': convert_state_dict_type(optimizer.state_dict()),
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)



def load_mix_model_state(model,enc_filename, dec_filename, enckey, deckey, newkey):

    if not os.path.exists(enc_filename):
        return None, [], None
    if not os.path.exists(dec_filename):
        return None, [], None

    enc_state = torch.load(enc_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    enc_state = _upgrade_state_dict(enc_state)


    dec_state = torch.load(dec_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    dec_state = _upgrade_state_dict(dec_state)

    enc_dict = {k.replace(enckey,newkey):v for k,v in dec_state['model'].items() if enckey + '.' + 'encoder' in k}
    dec_dict = {k.replace(deckey,newkey):v for k,v in dec_state['model'].items() if deckey + '.' + 'decoder' in k}
    enc_dict.update(dec_dict)
    state_model = OrderedDict(enc_dict)

    model.upgrade_state_dict(state_model)

    # load model parameters
    try:
        model.load_state_dict(state_model, strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return None, [], None

def load_pretrained_embeddings(path):
    with open(path) as f:
        tokens, dim = f.readline().split()
        #Init weights reserving first possitions for <Lua heritage> pad </s> <unk>
        weights = np.empty((int(tokens)+4,int(dim)),dtype='float')
        weights[1] = np.zeros(int(dim),dtype='float')
        i = 4
        for l in f.readlines():
            word, vec = l.split(' ',1)
            weights[i] = np.fromstring(vec,dtype='float')
            i += 1
        return weights


def get_nli_encoder(model,key,filename,tag):
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    current_state = model.state_dict()
    if not key is None:
        state['model'] = OrderedDict({k.replace('encoder',tag):v for k,v in state['model'].items() if '.'.join(['models',key,'encoder']) in k})
        state['model'] = OrderedDict({k.replace('models.' + key + '.',''):v for k,v in state['model'].items()})
    else:
        state['model'] = OrderedDict({k.replace('encoder',tag):v for k,v in state['model'].items() if 'encoder.' in k})
    for k,v in state['model'].items():
        current_state[k] = v
    model.load_state_dict(current_state,strict=False)
    model.upgrade_state_dict(state['model'])


def load_partial_unsup_model_state(enc_filename,dec_filename,pivot_enc_filename,pivot_dec_filename,model,keys,pivotkeys,newkeys,path=None):
    if not os.path.exists(enc_filename) or not os.path.exists(dec_filename) or not os.path.exists(pivot_enc_filename) or not os.path.exists(pivot_dec_filename):
        return None, [], None
    enc_state = torch.load(enc_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    enc_state = _upgrade_state_dict(enc_state)

    dec_state = torch.load(dec_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    dec_state = _upgrade_state_dict(dec_state)

    pivot_enc_state = torch.load(pivot_enc_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    pivot_enc_state = _upgrade_state_dict(pivot_enc_state)

    pivot_dec_state = torch.load(pivot_dec_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    pivot_dec_state = _upgrade_state_dict(pivot_dec_state)
    
    model_state = OrderedDict()
    pivot_model_state = OrderedDict()
    for key, newkey, pivotkey in zip(keys,newkeys,pivotkeys):
        rev_key = key.split('-')[1] + '-' + key.split('-')[0]
        model_state.update(OrderedDict({k.replace(key,newkey):v for k,v in enc_state['model'].items() if key + '.encoder' in k}))
        print('Key', key, 'Rev_key', rev_key)
        print('Model state after loading encoder', len(model_state))
        model_state.update(OrderedDict({k.replace(rev_key,newkey):v for k,v in dec_state['model'].items() if rev_key + '.decoder' in k}))
        print('Model state after loading decoder', len(model_state))
        rev_pkey = pivotkey.split('-')[1] + '-' + pivotkey.split('-')[0]
        pivot_model_state.update(OrderedDict({k.replace(pivotkey + '.encoder',newkey + '.pivot_encoder'):v for k,v in pivot_enc_state['model'].items() if pivotkey + '.encoder' in k}))
        pivot_model_state.update(OrderedDict({k.replace(rev_pkey + '.decoder',newkey + '.pivot_decoder'):v for k,v in pivot_dec_state['model'].items() if rev_pkey + '.decoder' in k}))
    
    print('Original enc state', len(enc_state['model']))
    print('Model state params', len(model_state))
    print('Pivot state params', len(pivot_model_state))
    model_state.update(pivot_model_state)
    enc_state['model'] = model_state

    model.load_state_dict(enc_state['model'], strict=False)
    # load model parameters
    try:
        model.load_state_dict(enc_state['model'], strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return enc_state['extra_state'], enc_state['optimizer_history'], enc_state['last_optimizer_state']


def load_partial_audio_model_state(enc_files,dec_files, model,enckeys,deckeys,newkeys,reuse,path=None):
    model_state = OrderedDict()
    enc_state = None
    for enc_file, dec_file, enckey, deckey,newkey in zip(enc_files, dec_files, enckeys, deckeys, newkeys):
        if not os.path.exists(enc_file) or not os.path.exists(dec_file):
            return None, [], None
        enc_state = torch.load(enc_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        enc_state = _upgrade_state_dict(enc_state)
    
        dec_state = torch.load(dec_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        dec_state = _upgrade_state_dict(dec_state)
    
        if reuse in ['encoder','both']:
            model_state.update(OrderedDict({k.replace(enckey,newkey):v for k,v in enc_state['model'].items() if enckey + '.encoder' in k}))
            print('Encoder params', enc_file, len(model_state),enckey,newkey)
        if reuse in ['decoder','both']:
            model_state.update(OrderedDict({k.replace(deckey,newkey):v for k,v in dec_state['model'].items() if deckey + '.decoder' in k}))
            print('Decoder params', dec_file,  len(model_state),deckey,newkey)
    
    enc_state['model'] = OrderedDict(model_state)
    print('LOAD PARTIAL MODEL', enckeys, deckeys ,newkeys,reuse, len(enc_state['model']))
    model.upgrade_state_dict(enc_state['model'])
    model_keys =  list(model.state_dict().keys())
    print('keys in model not in checkpoint')
    
    for k in model_keys:
        if k not in list(enc_state['model'].keys()):
            print(k)
    print('******************************')
    print('keys in checkpoint not in model')
    for k in  list(enc_state['model'].keys()):
        if k not in model_keys:
            print(k)
    print('******************************')
    # load model parameters
    try:
        model.load_state_dict(enc_state['model'], strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')


    return enc_state['extra_state'], enc_state['optimizer_history'], enc_state['last_optimizer_state']


def load_partial_model_state(filename, model,key,newkey,reuse,finetune,path=None):
    if not os.path.exists(filename):
        return None, [], None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = _upgrade_state_dict(state)
    if not finetune:
        state['model'] = OrderedDict({k.replace(key,newkey):v for k,v in state['model'].items() if key + '.' + reuse in k})
    else:
        state['model'] = OrderedDict({k.replace(key,newkey):v for k,v in state['model'].items()})
        if path:
            state['model']['.'.join(['model',newkey,reuse,'embed_tokens','weight'])] = load_pretrained_embeddings(path)
    print('LOAD PARTIAL MODEL', key,newkey,reuse,finetune, len(state['model']))
    model.upgrade_state_dict(state['model'])
    model_keys = [ k  for k in list(model.state_dict().keys()) if 'decoder' in k]
    print('keys in model not in checkpoint')
    for k in model_keys:
        if k not in list(state['model'].keys()):
            print(k)
    print('******************************')
    print('keys in checkpoint not in model')
    for k in  list(state['model'].keys()):
        if k not in model_keys:
            print(k)
    print('******************************')

    # load model parameters
    try:
        model.load_state_dict(state['model'], strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def load_model_state(filename, model):
    if not os.path.exists(filename):
        return None, [], None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = _upgrade_state_dict(state)
    model.upgrade_state_dict(state['model'])

    # load model parameters
    try:
        model.load_state_dict(state['model'], strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': 'CrossEntropyCriterion',
                'best_loss': state['best_loss'],
            },
        ]
        state['last_optimizer_state'] = state['optimizer']
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    # reduce optimizer history's memory usage (only keep the last state)
    if 'optimizer' in state['optimizer_history'][-1]:
        state['last_optimizer_state'] = state['optimizer_history'][-1]['optimizer']
        for optim_hist in state['optimizer_history']:
            del optim_hist['optimizer']
    # record the optimizer class name
    if 'optimizer_name' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['optimizer_name'] = 'FairseqNAG'
    # move best_loss into lr_scheduler_state
    if 'lr_scheduler_state' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['lr_scheduler_state'] = {
            'best': state['optimizer_history'][-1]['best_loss'],
        }
        del state['optimizer_history'][-1]['best_loss']
    # keep track of number of updates
    if 'num_updates' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['num_updates'] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state['args'], 'max_positions') and not hasattr(state['args'], 'max_source_positions'):
        state['args'].max_source_positions = state['args'].max_positions
        state['args'].max_target_positions = state['args'].max_positions
    # use stateful training data iterator
    if 'train_iterator' not in state['extra_state']:
        state['extra_state']['train_iterator'] = {
            'epoch': state['extra_state']['epoch'],
            'iterations_in_epoch': state['extra_state'].get('batch_offset', 0),
        }
    return state


def load_partial_model_for_inference(enc_filename,enc_key, dec_filename, dec_key, newkey, newarch,newtask,task,model_arg_overrides=None,pair=None):

    if not os.path.exists(enc_filename):
            raise IOError('Model file not found: {}'.format(enc_filename))
    enc_state = torch.load(enc_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    enc_state = _upgrade_state_dict(enc_state)
    enc_state['model'] = OrderedDict({k.replace('.'+enc_key+'.','.'+newkey+'.'):v for k,v in enc_state['model'].items() if enc_key + '.' + 'encoder' in k})
    print('Encoder params', len(enc_state['model']))
    if not os.path.exists(dec_filename):
            raise IOError('Model file not found: {}'.format(dec_filename))
    dec_state = torch.load(dec_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    dec_state = _upgrade_state_dict(dec_state)
    dec_state['model'] = OrderedDict({k.replace('.'+dec_key+'.','.'+newkey+'.'):v for k,v in dec_state['model'].items() if dec_key + '.' + 'decoder' in k})

    args = enc_state['args']
    args.task = newtask
    args.arch = newarch
    args.source_lang = newkey.split('-')[0]
    args.target_lang = newkey.split('-')[1]
    args.lang_pairs = [newkey]

    if model_arg_overrides is not None:
        args = _override_model_args(args, model_arg_overrides)

    enc_state['model'].update(dec_state['model'])
    state = enc_state['model']
    model = task.build_model(args)
    model.upgrade_state_dict(state)

    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    model.eval()

    return [model],args


def get_nli_encoder_state(task,enc_path,enc_key,tag):
    #load encoder state from previous model
    enc_state = torch.load(enc_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    print("ENC KEY:", enc_key)
    if not enc_key is None:
        enc_state['model'] = OrderedDict({k.replace('encoder',tag):v for k,v in enc_state['model'].items() if '.'.join(['models',enc_key,'encoder']) in k})
        enc_state['model'] = OrderedDict({k.replace('models.' + enc_key + '.',''):v for k,v in enc_state['model'].items()})
    else:
        enc_state['model'] = OrderedDict({k.replace('encoder',tag):v for k,v in enc_state['model'].items() if 'encoder.' in k})
    return enc_state

def load_nli_model_for_inference(path,task, model_arg_overrides=None,pair=None):

    ref_enc_path, hyp_enc_path = task.encoder_paths()
    ref_enc_key, hyp_enc_key = task.encoder_keys()

    if not os.path.exists(path):
            raise IOError('Model file not found: {}'.format(path))
    if not os.path.exists(ref_enc_path):
        raise IOError('Model file not found: {}'.format(ref_enc_path))
    if not os.path.exists(hyp_enc_path):
        raise IOError('Model file not found: {}'.format(hyp_enc_path))

    ref_enc_state = get_nli_encoder_state(task,ref_enc_path,ref_enc_key,'ref_encoder')
    hyp_enc_state = get_nli_encoder_state(task,hyp_enc_path,hyp_enc_key,'hyp_encoder')

    #Load classifier weights
    state = torch.load(path, map_location=lambda s, l: default_restore_location(s, 'cpu'))

    state['model'] = OrderedDict({k:v for k,v in state['model'].items() if 'classifier' in k})

    #Join models
    state['model'].update(ref_enc_state['model'])
    state['model'].update(hyp_enc_state['model'])

    state = _upgrade_state_dict(state)
    args = state['args']

    model = task.build_model(args)
    model.load_state_dict(state['model'], strict=False)
    model.upgrade_state_dict(state['model'])

    return model

def load_ensemble_for_inference(filenames, task, model_arg_overrides=None,pair=None):
    """Load an ensemble of models for inference.

    model_arg_overrides allows you to pass a dictionary model_arg_overrides --
    {'arg_name': arg} -- to override model args that were used during model
    training
    """
    # load model architectures and weights
    states = []
    for filename in filenames:
        print('loading new filename', filename)
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        state = _upgrade_state_dict(state)
        print(list(state['model'].keys()))
        if pair:
            state['model'] = OrderedDict({k:v for k,v in state['model'].items() if pair in k})
        print(list(state['model'].keys()))
        states.append(state)

    ensemble = []
    for state in states:
        args = state['args']

        if model_arg_overrides is not None:
            args = _override_model_args(args, model_arg_overrides)

        if pair:
            args.lang_pairs = [pair]

        # build model for ensemble
        model = task.build_model(args)
        model.upgrade_state_dict(state['model'])
        model.load_state_dict(state['model'], strict=True)
        ensemble.append(model)

        # some args (e.g., tokens_per_sample) might have been updated while building the model
        if model_arg_overrides is not None:
            args = _override_model_args(args, model_arg_overrides)

    return ensemble, args


def _override_model_args(args, model_arg_overrides):
    # Uses model_arg_overrides {'arg_name': arg} to override model args
    for arg_name, arg_val in model_arg_overrides.items():
        setattr(args, arg_name, arg_val)
    return args


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str):
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("| Found {}/{} types in embedding file.".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer
    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe):
    from fairseq import tokenizer
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tokenizer.Tokenizer.tokenize(hypo_str, tgt_dict, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx, left_pad, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    if onnx_trace:
        range_buf = torch._dim_arange(like=tensor, dim=1) + padding_idx + 1
        mask = tensor.ne(padding_idx)
        positions = range_buf.expand_as(tensor)
        if left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return positions * mask.long() + padding_idx * (1 - mask.long())

    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            else:
                max_positions = tuple(
                    map(nullsafe_min, zip(max_positions, arg))
                )
    return max_positions
