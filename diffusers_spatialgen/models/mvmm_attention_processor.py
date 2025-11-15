from typing import Callable, Optional

from einops import rearrange

from diffusers.models.attention_processor import *
from diffusers.utils import logging

logger = logging.get_logger(__name__)

# Copied from diffusers.models.attention_processor.JointAttnProcessor2_0
# The only modifications: reshape qkv by `num_views`
class JointMVAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        num_views: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Multi-view Self-Attn
        num_views = num_views if num_views is not None else kwargs.get("num_views", 1)
        key = rearrange(key, "(b v) n d -> b (v n) d", v=num_views)
        value = rearrange(value, "(b v) n d -> b (v n) d", v=num_views)
        query = rearrange(query, "(b v) n d -> b (v n) d", v=num_views)
        batch_size //= num_views

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]*num_views],
                hidden_states[:, residual.shape[1]*num_views :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # Multi-view Self-Attn
        hidden_states = rearrange(hidden_states, "b (v n) d -> (b v) n d", v=num_views)
        batch_size *= num_views

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


# Copied from diffusers.models.attention_processor.AttnProcessor2_0
class XFormersMVAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_input_views=4,
        num_output_views=4,
        multiview_attention=True,
        sparse_mv_attention=False,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        num_all_views = num_input_views + num_output_views

        # attn_type = "self-attn" if encoder_hidden_states is None else "cross-attn"
        # logger.info(f"[XFormersMVAttnProcessor]:  {attn_type} input hidden states {hidden_states.shape}")

        batch_size, key_tokens, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if multiview_attention:
            query = rearrange(query, "(b v) l d -> b (v l) d", v=num_all_views)
            key = rearrange(key, "(b v) l d -> b (v l) d", v=num_all_views)
            value = rearrange(value, "(b v) l d -> b (v l) d", v=num_all_views)
        # logger.info(f"[XFormersMVAttnProcessor]: {attn_type} query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
            
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        # logger.info(f"[XFormersMVAttnProcessor]: {attn_type} reshaped query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

        hidden_states = xformers.ops.memory_efficient_attention(  # query: (bm) l c -> b (ml) c;  key: b (nl) c
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        # logger.info(f"[XFormersMVAttnProcessor]: {attn_type} attented hidden_states shape: {hidden_states.shape}")
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # reshape back
        if multiview_attention:
            hidden_states = rearrange(hidden_states, "b (v l) d -> (b v) l d", v=num_all_views).contiguous()

        # logger.info(f"[XFormersMVAttnProcessor]:  {attn_type} output hidden states {hidden_states.shape}")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class XFormersJointAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=2,
        num_input_views=4,
        num_output_views=4,
    ):

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        num_all_views = num_input_views + num_output_views
        # logger.info(f"[XFormersJointAttnProcessor]: input hidden states {hidden_states.shape}")

        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # reshape hidden states for multiview & multimodal attention
        if num_tasks > 1:
            # Method3: modalral-attn: only fuse different modal tokens
            query = rearrange(query, "(b n_t v) l d ->  (b v) (n_t l) d", n_t=num_tasks, v=num_all_views)
            key = rearrange(key, "(b n_t v) l d ->  (b v) (n_t l) d", n_t=num_tasks, v=num_all_views)
            value = rearrange(value, "(b n_t v) l d ->  (b v) (n_t l) d", n_t=num_tasks, v=num_all_views)
        # logger.info(f"[XFormersJointAttnProcessor]: query {query.shape}, key {key.shape}, value {value.shape}")

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        # logger.info(f"[XFormersJointAttnProcessor]: reshaped query {query.shape}, key {key.shape}, value {value.shape}")

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        # logger.info(f"[XFormersJointAttnProcessor]: score : {hidden_states.shape}")

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # reshape back
        if num_tasks > 1:
            # Method3: modal-attn: only fuse different modal tokens
            hidden_states = rearrange(hidden_states, "(b v) (n_t l) d -> (b n_t v) l d", n_t=num_tasks, v=num_all_views)
        # logger.info(f"[XFormersJointAttnProcessor]: hiidden states {hidden_states.shape}")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
