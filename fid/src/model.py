# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import CrossEntropyLoss
from einops import einsum, rearrange, repeat
from typing import TypeVar, overload
from copy import deepcopy

from .modeling_t5 import T5Attention
from .modeling_t5 import T5ForConditionalGeneration
from .modeling_encoder_decoder import EncoderDecoderModel

class FiDO_T5Attention(nn.Module):
    def __init__(self, 
                 is_decoder: bool,
                 d_model: int,
                 key_value_proj_dim: int,
                 n_heads: int,
                 kv_heads: int,         # GQA할 때 K, V head의 개수
                 dropout: float,
                 has_relative_attention_bias: bool,
                 relative_attention_num_buckets: int,
                 relative_attention_max_distance: int,
    ):
        super().__init__()

        if n_heads % kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by kv_heads ({kv_heads})"
            )
        
        self.is_decoder = is_decoder
        self.d_model = d_model
        self.key_value_proj_dim = key_value_proj_dim
        self.n_heads = n_heads

        self.kv_heads = kv_heads                                # GQA
        self.dropout = dropout
                
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.kv_dim = self.kv_heads * self.key_value_proj_dim   # GQA

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

        self._relative_position_bucket = T5Attention._relative_position_bucket

    @classmethod
    def from_t5_attention(cls, 
                          t5: T5Attention, 
                          kv_heads: int):
        FiDO_t5 = FiDO_T5Attention(
            is_decoder = t5.is_decoder,
            d_model = t5.d_model,
            key_value_proj_dim = t5.key_value_proj_dim,
            n_heads = t5.n_heads,
            kv_heads = kv_heads,            # GQA
            dropout = t5.dropout,
            has_relative_attention_bias = t5.has_relative_attention_bias,
            relative_attention_num_buckets = t5.relative_attention_num_buckets,
            relative_attention_max_distance = t5.relative_attention_max_distance,
        )

        FiDO_t5.q.weight.data = t5.q.weight.data
        FiDO_t5.k.weight.data = t5.k.weight.data
        FiDO_t5.v.weight.data = t5.v.weight.data
        FiDO_t5.o.weight.data = t5.o.weight.data

        if t5.has_relative_attention_bias:
            FiDO_t5.relative_attention_bias.weight.data = (t5.relative_attention_bias.weight.data)
        
        return FiDO_t5

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        
        real_seq_length = seq_length
        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            # query 길이만큼 Length를 더해줌
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        # key_value_states: batch_size x (n_passages * seq_len) x hidden_dim                
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
        
        # Maybe: ( n_passages * seq_len )
        def shape(states):
            """projection"""
            sequence_length = states.shape[1]
            return states.view(batch_size, 
                               sequence_length, 
                               -1, #self.n_heads, 
                               self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            sequence_length = states.shape[2]
            return (states.transpose(1, 2).contiguous().view(batch_size, 
                                                             sequence_length,#-1, 
                                                             -1))#self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            # 동일할 경우, past_key_value로 치환하는 것 같은데?!!! => 그래도 되는건가 ?
            if past_key_value is not None:
                # print("block_num이 1부터 여기 통과하지?!!! block_num:", self.block_num)
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        # NOTE: GQA 적용
        # query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        grouped_queries = shape(self.q(hidden_states))

        # get key/value states        
        # NOTE: 문제가 생긴다면 past_key_value 일수도?
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        # scores = torch.matmul(
        #     query_states, key_states.transpose(3, 2)
        # )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        grouped_queries = rearrange(grouped_queries,
                                    "b (g h) n d -> b g h n d",
                                    h = self.kv_heads)
        grouped_keys = rearrange(key_states,
                                 "b (g h) s d -> b g h s d",
                                 h = self.kv_heads).mean(dim=1)
        grouped_values = rearrange(value_states,
                                   "b (g h) s d -> b g h s d",
                                   h = self.kv_heads).mean(dim=1)
        scores = einsum(grouped_queries,
                        grouped_keys,
                        "b g h n d, b h s d -> b h n s")
        
        # 맨 처음에만 없고, 그 이후에는 존재함 ( decoder에서 )
        if position_bias is None:
            if not self.has_relative_attention_bias:
                # NOTE: GQA 적용
                # position_bias = torch.zeros(
                #     (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                # )
                position_bias = torch.zeros(
                    (1, self.kv_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
                
            else:
                # NOTE: GQA 적용
                # position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
                position_bias = T5Attention.compute_bias(self, real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # NOTE: GQA 적용
        grouped_position_bias = rearrange(position_bias_masked,
                                          "b (g h) n s -> b g h n s",
                                          h = self.kv_heads).mean(dim=1)
        
        # scores += position_bias_masked
        scores += grouped_position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # NOTE: GQA 적용
        # attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = unshape(torch.matmul(attn_weights, grouped_values))
        attn_output = repeat(
            attn_output, 
            "b s d -> b s (g d)",
            g = (self.n_heads // self.kv_heads)
        ) # 줄인만큼 다시 늘려주기
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

ModuleType = TypeVar("ModuleType", bound=nn.Module)

@overload
def convert_GQA(
    module: ModuleType, kv_heads: int, inplace: bool = False
) -> ModuleType:
    ...

@overload
def convert_GQA(
    module: T5Attention, kv_heads: int, inplace: bool = False
) -> FiDO_T5Attention:
    ...

def convert_GQA(module, 
                       kv_heads: int,
                       inplace: bool = False):
    if isinstance(module, T5Attention):
        return FiDO_T5Attention.from_t5_attention(module, 
                                                  kv_heads=kv_heads)

    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        out._modules[name] = convert_GQA(child,
                                                kv_heads=kv_heads,
                                                inplace=True)
    return out

def convert_LSA(module,
                n_cross_layer: int = None,
                block = 't5'):
    
            # if n_layer and n_cross_layer: # LSA 적용할 경우에만 실행
        #     if n_layer % n_cross_layer != 0:
        #         raise ValueError(
        #             f"n_layer ({n_layer}) must be divisible by n_cross_layer ({n_cross_layer})"
        #         )
    if n_cross_layer:

        if block == 'skt':
            decoder_block_count = len(module.decoder.decoder.transformer.h)
            for i in range(decoder_block_count):
                module.decoder.decoder.transformer.h[i].is_decoder = True
                if (i+1) % n_cross_layer != 0:
                    module.decoder.decoder.transformer.h[i]._modules.pop('crossattention')
                    module.decoder.decoder.transformer.h[i]._modules.pop('ln_cross_attn')
                    module.decoder.decoder.transformer.h[i].is_decoder = False
        elif block == 't5':
            decoder_block_count = len(module.decoder.decoder.block)
            for i in range(decoder_block_count):
                if (i+1) % n_cross_layer != 0:
                    block = module.decoder.decoder.block[i].layer
                    block = nn.ModuleList([block[0], block[2]])
                    module.decoder.decoder.block[i].layer = block
                    module.decoder.decoder.block[i].is_decoder = False
        else:
            ValueError("choose between 't5' and 'skt'")
    
    return module

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config, first_k=None):
        super().__init__(config)
        self.first_k = first_k
        self.wrap_encoder()
        self.wrap_decoder() # first_k가 있을 경우 적용

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):

        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                # self.decoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        # T5ForConditionalGeneration의 Decoder 단에서
        # decoder_input_ids       => label에 대한 ids               # b x label_length
        # decoder_attention_mask  => label에 대한 attention mask    # b x label_length
        # encoder_hidden_states   => encoder를 통과한 hidden_states # b x (n_passage * seq_len) x hidden_dim )
        # encoder_attention_mask  => context에 대한 attention mask  # b x (n_passage * seq_len)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)

        # input_ids => batch_size x n_passages x seq_len
        # view 적용 => batch_size x (n_passages * seq_len)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_decoder(self):
        self.decoder = DecoderWrapper(self.decoder,
                                      first_k=self.first_k)
    
    def unwrap_decoder(self):
        self.decoder = self.decoder.decoder
        block = []
        for mod in self.decoder.block:
            block.append(mod) # checkpointer 사용 할 경우 `mod.module` 사용
        block = nn.ModuleList(block)
        self.decoder.block = block

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder,
                                      use_checkpoint=use_checkpoint,
                                      first_k=self.first_k)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module) 
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.unwrap_decoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
        self.wrap_decoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class DecoderWrapper(torch.nn.Module):
    """
        Decoder에 들어오는 context attention mask를 Token 길이로 변환
    """
    def __init__(self, decoder, first_k=None):
        super().__init__()
        self.decoder = decoder
        self.first_k = first_k
    
    def forward(
        self,
        input_ids=None, # decoder_input_ids: label
        attention_mask=None, # 없음
        inputs_embeds=None,
        past_key_values=None,
        encoder_hidden_states=None, # encoder_hidden_states
        encoder_attention_mask=None, # input attention mask
        **kwargs,
    ):
        if self.first_k:
            nk = encoder_hidden_states.size(1) # n_passages * first_k
            n_passages = nk // self.first_k
            
            # b x (n_passages * seq_len) -> b x n_passages x seq_len -> b x n_passages x first_k -> b x (n_passages * first_k)
            encoder_attention_mask = encoder_attention_mask.reshape(encoder_attention_mask.size(0), n_passages, -1)
            encoder_attention_mask = encoder_attention_mask[:, :, :self.first_k]
            encoder_attention_mask = encoder_attention_mask.reshape(encoder_attention_mask.size(0), -1)
            
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, # 이거때문..? ====> past key value가 없어서 일단 느려지는 게 아닐까??
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, 
            **kwargs,
        )
        return outputs
        
class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False, first_k=None, skt=False):
        super().__init__()

        # AttributeError: 'EncoderWrapper' object has no attribute 'main_input_name'
        # 위 에러로 인하여 수정 ( Github Issue에서 발견 )
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        self.first_k = first_k
        if skt:
            apply_checkpoint_wrapper_SKT(self.encoder, use_checkpoint)
        else:
            apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs) # (bsz * n_passages) x seq_len x hidden_dim
        # return outputs
        if self.first_k:
            last_hidden_state = outputs[0][:, :self.first_k, :] # bsz x first_k Token x Hidden_dim
            last_hidden_state = last_hidden_state.reshape(bsz, self.n_passages * self.first_k, -1)
            outputs = (last_hidden_state,) + outputs[1:]
        else:
            outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        # outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper_SKT(t5stack, use_checkpoint):
    block = []
    for mod in t5stack.encoder.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.encoder.block = block

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class FiDSKT(EncoderDecoderModel):
    def __init__(self, config, encoder_model_name, first_k=None):
        super().__init__(config)
        # self.encoder = transformers.T5EncoderModel.from_pretrained('KETI-AIR/ke-t5-large') # opt.t5_model_name 으로 바꾸기
        self.encoder = transformers.T5EncoderModel.from_pretrained(encoder_model_name)
        
        self.first_k = first_k
        self.wrap_encoder()
        self.wrap_decoder() # first_k가 있을 경우 적용
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):

        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                # self.decoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        # T5ForConditionalGeneration의 Decoder 단에서
        # decoder_input_ids       => label에 대한 ids               # b x label_length
        # decoder_attention_mask  => label에 대한 attention mask    # b x label_length
        # encoder_hidden_states   => encoder를 통과한 hidden_states # b x (n_passage * seq_len) x hidden_dim )
        # encoder_attention_mask  => context에 대한 attention mask  # b x (n_passage * seq_len)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)

        # input_ids => batch_size x n_passages x seq_len
        # view 적용 => batch_size x (n_passages * seq_len)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_decoder(self):
        self.decoder = DecoderWrapper(self.decoder,
                                      first_k=self.first_k)

    def unwrap_decoder(self):
        self.decoder = self.decoder.decoder
        h = []
        for mod in self.decoder.transformer.h:
            h.append(mod)
        h = nn.ModuleList(h)
        self.decoder.transformer.h = h

    def wrap_encoder(self, use_checkpoint=False):
        self.encoder = EncoderWrapper(self.encoder,
                                      use_checkpoint=use_checkpoint,
                                      first_k=self.first_k,
                                      skt=True)
    
    def unwrap_encoder(self):
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.encoder.block:
            block.append(mod.module)    
        block = nn.ModuleList(block)
        self.encoder.encoder.block = block

    def load_encdec(self, state_dict):
        self.unwrap_encoder()
        self.unwrap_decoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
        self.wrap_decoder()
    
    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp