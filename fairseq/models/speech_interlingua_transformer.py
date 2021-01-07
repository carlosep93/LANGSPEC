# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import torch
from fairseq import utils
from fairseq.tasks.interlingua_translation import InterlinguaTranslationTask
from fairseq.tasks.dist_interlingua_translation import DistInterlinguaTranslationTask

from . import register_model, register_model_architecture
from fairseq.models.fairseq_model import FairseqInterlinguaModel
from .s_transformer import (
    base_architecture,
    Embedding,
    SpeechTransformerModel,
    SpeechTransformerEncoder,
    SpeechTransformerDecoder,
)


@register_model('speech_interlingua_transformer')
class InterlinguaTransformerModel(FairseqInterlinguaModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders,keys,auto=True):
        super().__init__(encoders, decoders,keys,auto)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        SpeechTransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--tie-lang-embeddings', action='store_true',
                            help='tie embedding table by language')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        #assert isinstance(task, InterlinguaTranslationTask) or isinstance(task, DistInterlinguaTranslationTask)

        # make sure all arguments are present in older models
        base_interlingua_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        print('*******************')
        print('LANG PAIRS', args.lang_pairs)
        print('*******************')
        src_langs = [lang_pair.split('-')[0] for lang_pair in args.lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in args.lang_pairs]


        lang_embeddings = {}


        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            shared_dict = task.dicts[task.langs[0]]
            if any(dict != shared_dict for dict in task.dicts.values()):
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = build_embedding(
                shared_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_dict = task.dicts[src_langs[0]]
                if any(task.dicts[src_lang] != shared_dict for src_lang in src_langs):
                    raise ValueError('--share-encoder-embeddings requires a joined source dictionary')
                shared_encoder_embed_tokens = build_embedding(
                    shared_dict, args.encoder_embed_dim, args.encoder_embed_path
                )
            if args.share_decoder_embeddings:
                shared_dict = task.dicts[tgt_langs[0]]
                if any(task.dicts[tgt_lang] != shared_dict for tgt_lang in src_langs):
                    raise ValueError('--share-decoder-embeddings requires a joined target dictionary')
                shared_decoder_embed_tokens = build_embedding(
                    shared_dict, args.decoder_embed_dim, args.decoder_embed_path
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(args,lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                elif args.tie_lang_embeddings and lang in lang_embeddings:
                    encoder_embed_tokens = lang_embeddings[lang]
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                    if args.tie_lang_embeddings:
                        lang_embeddings[lang] = encoder_embed_tokens
                pos = sorted(src_langs).index(lang)
                print("TASK AUDIO FEATURES", task.audio_features)
                lang_encoders[lang] =SpeechTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens,audio_features=task.audio_features).to('cuda:' + str(pos)) \
                                        if torch.cuda.device_count() > 1 else \
                                        SpeechTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens,audio_features=task.audio_features)
            return lang_encoders[lang]

        def get_decoder(args,lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                elif args.tie_lang_embeddings and lang in lang_embeddings:
                    decoder_embed_tokens = lang_embeddings[lang]
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                    if args.tie_lang_embeddings:
                        lang_embeddings[lang] = decoder_embed_tokens
                lang_decoders[lang] = SpeechTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens,final_norm=task.final_norm)
                #lang_decoders[lang] = SpeechTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens,final_norm=False)
            return lang_decoders[lang]


        def try_decoder(args,src,tgt):
            try:
                return shared_decoder if shared_decoder is not None else get_decoder(args,src)
            except KeyError:
                return shared_decoder if shared_decoder is not None else get_decoder(args,tgt)


        def try_encoder(args,tgt,src):
            try:
                return shared_encoder if shared_encoder is not None else get_encoder(args,tgt)
            except KeyError:
                return shared_encoder if shared_encoder is not None else get_encoder(args,src)

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])


        for lang_pair, src, tgt in zip(args.lang_pairs, src_langs, tgt_langs):
            lang_pair = lang_pair.split('-')
            lang_encoders[lang_pair[0]] = try_encoder(args,src,tgt)
            lang_decoders[lang_pair[1]] = try_decoder(args,tgt,src)

        auto = task.auto if hasattr(task,'auto') else False
        return InterlinguaTransformerModel(lang_encoders, lang_decoders,args.lang_pairs,auto)





@register_model_architecture('speech_interlingua_transformer', 'speech_interlingua_transformer_base')
def base_interlingua_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)
    args.tie_lang_embeddings = getattr(args,'tie_lang_embeddings',False)



@register_model_architecture('speech_interlingua_transformer', 'speech_interlingua_transformer_big')
def speechtransformer_fbk(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.adapter_netword = getattr(args, 'adapter_network', False)
    args.adapter_size = getattr(args, 'adapter_size', 1024)



@register_model_architecture('speech_interlingua_transformer', 'speech_interlingua_transformer_big_3conv')
def speechtransformer_fbk(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 3')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.adapter_netword = getattr(args, 'adapter_network', False)
    args.adapter_size = getattr(args, 'adapter_size', 1024)



