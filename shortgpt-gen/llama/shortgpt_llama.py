from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ShortgptLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(ShortgptLlamaModel, self).__init__(config)
        self.prune_layers = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        bsz, q_len, _ = hidden_states.size()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if hidden_states.shape[1] > 1 and self.prune_layers is not None and idx in self.prune_layers:
                if use_cache:
                    normed_hidden_states = decoder_layer.input_layernorm(hidden_states)

                    key_states = decoder_layer.self_attn.k_proj(normed_hidden_states)
                    value_states = decoder_layer.self_attn.v_proj(normed_hidden_states)

                    key_states = key_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)

                    if position_embeddings is None:
                        logger.warning_once(
                            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                            "removed and `position_embeddings` will be mandatory."
                        )
                        cos, sin = decoder_layer.self_attn.rotary_emb(value_states, position_ids)
                    else:
                        cos, sin = position_embeddings

                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    next_decoder_cache = past_key_values.update(key_states, value_states, idx, cache_kwargs)

                if output_attentions:
                    all_self_attns += (None,)

            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShortgptLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = ShortgptLlamaModel(config)
