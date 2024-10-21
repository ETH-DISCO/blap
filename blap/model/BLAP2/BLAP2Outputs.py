"""
Copyright (c) 2022 Salesforce, Inc.
All rights reserved.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

@dataclass
class BlapSimilarity(ModelOutput):
    sim_a2t: torch.FloatTensor = None
    sim_t2a: torch.FloatTensor = None

    sim_a2t_m: Optional[torch.FloatTensor] = None
    sim_t2a_m: Optional[torch.FloatTensor] = None

    sim_a2t_targets: Optional[torch.FloatTensor] = None
    sim_t2a_targets: Optional[torch.FloatTensor] = None


@dataclass
class BlapIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of Blap models.

    audio_embeds (torch.FloatTensor): audio embeddings, shape (batch_size, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    audio_embeds_m (torch.FloatTensor): audio embeddings from momentum visual encoder, shape (batch_size, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the audio-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the audio-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the audio-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    atm_logits (torch.FloatTensor): logits for the audio-text matching loss, shape (batch_size * 3, 2).
    atm_labels (torch.LongTensor): labels for the audio-text matching loss, shape (batch_size * 3,)

    """

    # uni-modal features
    audio_embeds: torch.FloatTensor = None
    audio_features: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None

    audio_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    atm_logits: Optional[torch.FloatTensor] = None
    atm_labels: Optional[torch.LongTensor] = None
    vl_embedding: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None

@dataclass
class BlapOutput(ModelOutput):
    # some finetuned models (e.g. BlapVQA) do not compute similarity, thus optional.
    sims: Optional[BlapSimilarity] = None

    intermediate_output: BlapIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_atc: Optional[torch.FloatTensor] = None

    loss_atm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None