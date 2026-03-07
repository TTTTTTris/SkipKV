import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging as hf_logging
import logging 
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import DynamicCache
from .compression import (
    SkipKV,
    R1KV,
    SnapKV,
    StreamingLLM,
    H2O,
    AnalysisKV
)

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

KV_COMPRESSION_MAP = {
    "skipkv": SkipKV,
    "rkv": R1KV,
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
    "analysiskv": AnalysisKV
}

logger = hf_logging.get_logger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# logging.getLogger().setLevel(logging.WARNING)

def LlamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    self.remove_punct_ranges = None
    self.sentence_similarity = None
    # =============== New logic end =================

def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        
        # Get completed punct ranges from the model if available
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'completed_punct_ranges'):
            completed_punct_ranges = self.config._parent_model.completed_punct_ranges
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'current_position'):
            current_position = self.config._parent_model.current_position
        
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                layer_idx=self.layer_idx,
                remove_punct_ranges=self.remove_punct_ranges,
                current_pos = current_position,
                padding_mask = attention_mask
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                layer_idx=self.layer_idx,
                remove_punct_ranges=self.remove_punct_ranges,
                current_pos = current_position,
                padding_mask = attention_mask
            )

            if self.config.update_kv is True:
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen2Attention_init(
    self, config: Qwen2Config, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    self.remove_punct_ranges = None
    self.sentence_similarity = None
    # =============== New logic end =================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end ===============

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        
        # Get completed punct ranges from the model if available
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'completed_punct_ranges'):
            completed_punct_ranges = self.config._parent_model.completed_punct_ranges
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'current_position'):
            current_position = self.config._parent_model.current_position
        
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                layer_idx=self.layer_idx,
                remove_punct_ranges=self.remove_punct_ranges,
                current_pos = current_position,
                padding_mask = attention_mask
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                remove_punct_ranges=self.remove_punct_ranges,
                layer_idx=self.layer_idx,
                current_pos = current_position,
                padding_mask = attention_mask
            )
            if self.config.update_kv is True:
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen3Attention_init(
    self, config: Qwen3Config, layer_idx: int, compression_config: dict
):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        # =============== New logic start ===============
        self.config.update(compression_config)
        self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
            **compression_config["method_config"]
        )
        self.remove_punct_ranges = None
        self.sentence_similarity = None
        # =============== New logic end =================

def Qwen3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        
        # Get completed punct ranges from the model if available
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'completed_punct_ranges'):
            completed_punct_ranges = self.config._parent_model.completed_punct_ranges
        if hasattr(self.config, '_parent_model') and hasattr(self.config._parent_model, 'current_position'):
            current_position = self.config._parent_model.current_position
        
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                remove_punct_ranges=self.remove_punct_ranges,
                layer_idx=self.layer_idx,
                current_pos = current_position,
                padding_mask = attention_mask
            )

            past_key_value.update(
                key_states_compress,
                value_states_compress,
                self.layer_idx,
                cache_kwargs,
            )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                sentence_similarity=self.sentence_similarity,
                completed_punct_ranges=completed_punct_ranges,
                remove_punct_ranges=self.remove_punct_ranges,
                layer_idx=self.layer_idx,
                current_pos = current_position,
                padding_mask = attention_mask
            )

            past_key_value.key_cache[self.layer_idx] = key_states_compress
            past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def set_steering_flag(self, steering_flag, steering_layer=None, steer_vec=None,  steer_coef=0.0, steer_gamma=0.0, tokenizer=None):
    self.steering_flag = steering_flag
    self.steering_vector = steer_vec
    self.steering_layer = steering_layer
    self.steering_coef = torch.tensor(steer_coef)
    self.steering_gamma = torch.tensor(steer_gamma)
    self.steering_coef_init = torch.tensor(steer_coef)
    self.steering_think_flag=None
    self.steering_split_ids = None
    self.steering_think_start_id=None
    self.steering_think_end_id=None
    if steering_flag:
        assert steering_layer is not None, "Steering layer must be provided for steering"
        assert steer_vec is not None, "Steering vector must be provided for steering"
        assert tokenizer is not None, "Tokenizer must be provided for steering"
        vocab = tokenizer.get_vocab()
        self.steering_split_ids = torch.LongTensor([vocab[token] for token in vocab.keys() if "ĊĊ" in token]).to(self.device)
        self.steering_think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.steering_think_end_id =  tokenizer.encode("</think>", add_special_tokens=False)[0]

def start_new_round(self, steer_coef):
    self.new_round=True
    self.cur_steps = 0
    self.steering_think_flag=None
    self.steering_coef = torch.tensor(steer_coef)
    self.steering_coef_init = torch.tensor(steer_coef)

def CausalLM_init(self, config):
    # Detect model type from config and use appropriate base class and model
    if isinstance(config, LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    elif isinstance(config, Qwen2Config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    elif isinstance(config, Qwen3Config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}. Supported types are LlamaConfig and Qwen2Config.")
    
    self.vocab_size = config.vocab_size

    self.new_round=False
    self.cur_steps = 0

    self.steering_flag = False
    self.steering_vector = None
    self.steering_layer = None
    self.steering_coef = torch.tensor(0.0)
    self.steering_gamma = torch.tensor(0.0)
    self.steering_coef_init = torch.tensor(0.0)
    self.steering_think_flag=None

    self.steering_split_ids = None
    self.steering_think_start_id=None
    self.steering_think_end_id=None

    # Initialize wait token detection attributes
    self.monitor_punct_tokens = False
    self.check_token_ids = []
    self.punct_token_ids = []
    self.current_position = 0
    self.punct_token_positions = []
    self.completed_punct_ranges = []  # List of tuples: (start_pos, punct_pos) for sentence segmentation
    self.last_sentence_start = 0  # Track sentence start position
    self.labeled_sentence_ranges = []
    self.label = {}
    self.num_sentences = []
    self._accumulated_hidden_states = None  # Keep for current step processing
    
    # Set reference to this model for attention layers to access wait ranges
    config._parent_model = self
    
    # Initialize weights and apply final processing
    self.post_init()

def Qwen2ForCausalLM_init(self, config):
    super(Qwen2ForCausalLM, self).__init__(config)
    self.model = Qwen2Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize wait token detection attributes
    self.monitor_punct_tokens = False
    self.check_token_ids = []
    self.punct_token_ids = []
    self.current_position = 0
    self.punct_token_positions = []
    self.completed_punct_ranges = []  # List of tuples: (start_pos, punct_pos) for sentence segmentation
    self.last_sentence_start = 0  # Track sentence start position
    self.labeled_sentence_ranges = []
    self.label = {}
    self.num_sentences = []
    self._accumulated_hidden_states = None  # Keep for current step processing
    # Set reference to this model for attention layers to access wait ranges
    config._parent_model = self

def Qwen3ForCausalLM_init(self, config):
    super(Qwen3ForCausalLM, self).__init__(config)
    self.model = Qwen3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize wait token detection attributes
    self.monitor_punct_tokens = False
    self.check_token_ids = []
    self.punct_token_ids = []
    self.current_position = 0
    self.punct_token_positions = []
    self.completed_punct_ranges = []  # List of tuples: (start_pos, punct_pos) for sentence segmentation
    self.last_sentence_start = 0  # Track sentence start position
    self.labeled_sentence_ranges = []
    self.label = {}
    self.num_sentences = []
    self._accumulated_hidden_states = None  # Keep for current step processing
    # Set reference to this model for attention layers to access wait ranges
    config._parent_model = self

def LlamaForCausalLM_init(self, config):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = LlamaModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize wait token detection attributes
    self.monitor_punct_tokens = False
    self.check_token_ids = []
    self.punct_token_ids = []
    self.current_position = 0
    self.punct_token_positions = []
    self.completed_punct_ranges = []  # List of tuples: (start_pos, punct_pos) for sentence segmentation
    self.last_sentence_start = 0  # Track sentence start position
    self.labeled_sentence_ranges = []
    self.label = {}
    self.num_sentences = []
    self._accumulated_hidden_states = None  # Keep for current step processing
    # Set reference to this model for attention layers to access wait ranges
    config._parent_model = self

def check_punct_tokens_in_forward(self, input_ids=None, padding_mask=None): 
    """
    Check if any punctuation tokens are encountered and record their positions for sentence segmentation.
    Also, tag sentence ranges with 'Reflection' if they include the 'wait' token.
    Supports multi-batch processing.
    
    Args:
        input_ids: Current input token IDs (optional, for direct token checking) - shape: [batch_size, seq_len]
        tokenizer: Tokenizer for decoding tokens (optional, for visualization)
    """       

    if input_ids is not None:
        batch_size, seq_len = input_ids.shape
        
        # Initialize per-batch tracking if not exists or empty
        if not hasattr(self, 'current_position_per_batch') or len(self.current_position_per_batch) != batch_size:
            self.current_position_per_batch = [0] * batch_size
        if not hasattr(self, 'punct_token_positions_per_batch') or len(self.punct_token_positions_per_batch) != batch_size:
            self.punct_token_positions_per_batch = [[] for _ in range(batch_size)]
        if not hasattr(self, 'completed_punct_ranges_per_batch') or len(self.completed_punct_ranges_per_batch) != batch_size:
            self.completed_punct_ranges_per_batch = [[] for _ in range(batch_size)]
        if not hasattr(self, 'last_sentence_start_per_batch') or len(self.last_sentence_start_per_batch) != batch_size:
            self.last_sentence_start_per_batch = [0] * batch_size
            # Skip token if it's padding
            if padding_mask is not None:
                for batch_idx in range(batch_size):
                    self.last_sentence_start_per_batch[batch_idx] = len(padding_mask[batch_idx]) - padding_mask[batch_idx].sum().item()

        if not hasattr(self, 'labeled_sentence_ranges_per_batch') or len(self.labeled_sentence_ranges_per_batch) != batch_size:
            self.labeled_sentence_ranges_per_batch = [[] for _ in range(batch_size)]
        if not hasattr(self, 'label_per_batch') or len(self.label_per_batch) != batch_size:
            self.label_per_batch = {i: [] for i in range(batch_size)}   
        if not hasattr(self, '_sentence_representations_per_batch') or len(self._sentence_representations_per_batch) != batch_size:
            self._sentence_representations_per_batch = [{} for _ in range(batch_size)]
        if not hasattr(self, '_remove_punct_ranges_per_batch') or len(self._remove_punct_ranges_per_batch) != batch_size:
            self._remove_punct_ranges_per_batch = [{} for _ in range(batch_size)]
        
        for batch_idx in range(batch_size):
            sample_tokens = input_ids[batch_idx]  # [seq_len]
            
            for i, token_id in enumerate(sample_tokens):   
                token_pos = self.current_position_per_batch[batch_idx] + i                 
                if token_id.item() in self.check_token_ids:
                    self.label_per_batch[batch_idx].append(token_pos)
                    # self.check_count[batch_idx] += 1
                    break

            for i, token_id in enumerate(sample_tokens):
                token_pos = self.current_position_per_batch[batch_idx] + i                 

                if token_id.item() in self.punct_token_ids:
                    self.punct_token_positions_per_batch[batch_idx].append(token_pos)
                    
                    # Create a sentence range from last sentence start to this punctuation
                    if self.last_sentence_start_per_batch[batch_idx] <= token_pos:
                        sentence_range = (self.last_sentence_start_per_batch[batch_idx], token_pos)
                    else:
                        # Extend last range
                        if len(self.completed_punct_ranges_per_batch[batch_idx]) > 0:
                            last_range = self.completed_punct_ranges_per_batch[batch_idx][-1]
                            sentence_range = (last_range[0], token_pos)
                            self.completed_punct_ranges_per_batch[batch_idx][-1] = sentence_range
                            # logging.info(f"Batch {batch_idx}: Extended last range to include consecutive punct: {sentence_range}")
                            self.last_sentence_start_per_batch[batch_idx] = token_pos + 1
                            continue
                        else:
                            sentence_range = (self.last_sentence_start_per_batch[batch_idx], token_pos)

                    self.completed_punct_ranges_per_batch[batch_idx].append(sentence_range)

                    self.last_sentence_start_per_batch[batch_idx] = token_pos + 1

                    if self.label_per_batch[batch_idx] and any(sentence_range[0] <= x <= sentence_range[-1] for x in self.label_per_batch[batch_idx]):
                            self.labeled_sentence_ranges_per_batch[batch_idx].append((sentence_range, 'Check'))
                    else:
                        self.labeled_sentence_ranges_per_batch[batch_idx].append((sentence_range, 'Others'))

            # Update position for this batch sample
            self.current_position_per_batch[batch_idx] += seq_len
            # self.num_sentences[batch_idx] = len(self.completed_punct_ranges_per_batch[batch_idx])

        self.current_position = self.current_position_per_batch
        self.punct_token_positions = self.punct_token_positions_per_batch
        self.completed_punct_ranges = self.completed_punct_ranges_per_batch
        self.labeled_sentence_ranges = self.labeled_sentence_ranges_per_batch
        self.label = self.label_per_batch


def reset_for_new_sample(self):
    """
    Reset all model state for a new sample. Call this between different samples
    to ensure clean state and consistent results.
    """
    # Reset position tracking and sentence/punctuation state
    if self.monitor_punct_tokens:
        self.current_position = 0
        self.punct_token_positions = []
        self.pending_wait_ranges = []  # Clear pending wait ranges
        self.completed_punct_ranges = []  # Clear completed punctuation ranges
        self.labeled_sentence_ranges = []  # Clear completed punctuation ranges
        self.label = {}
        self.num_sentences = []
        self._accumulated_hidden_states = None
        self._sentence_representations = []  # Clear sentence representations
        self.last_sentence_start = 0  # Reset sentence tracking 

        # Reset per-batch tracking attributes
        if hasattr(self, 'current_position_per_batch'):
            self.current_position_per_batch = []
        if hasattr(self, 'punct_token_positions_per_batch'):
            self.punct_token_positions_per_batch = []
        if hasattr(self, 'completed_punct_ranges_per_batch'):
            self.completed_punct_ranges_per_batch = []
        if hasattr(self, 'labeled_sentence_ranges_per_batch'):
            self.labeled_sentence_ranges_per_batch = []
        if hasattr(self, 'label_per_batch'):
            self.label_per_batch = {}
        if hasattr(self, 'last_sentence_start_per_batch'):
            self.last_sentence_start_per_batch = []
        if hasattr(self, '_sentence_representations_per_batch'):
            self._sentence_representations_per_batch = []
        if hasattr(self, '_remove_punct_ranges_per_batch'):
            self._remove_punct_ranges_per_batch = []
        if hasattr(self, "generated_token_ids"):
            self.generated_token_ids = []
        
        logging.info('Reset input position ids for new sample!')
    
    # Reset compression method's internal state for all layers
    if hasattr(self, 'model') and hasattr(self.model, 'layers'):
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'kv_cluster') and hasattr(layer.self_attn.kv_cluster, 'reset_for_new_batch'):
                layer.self_attn.kv_cluster.reset_for_new_batch()
    
    # Reset any model-level statistical tracking
    if hasattr(self, 'length'):
        self.length = 0
    if hasattr(self, 'cur_steps'):
        self.cur_steps = 0
    if hasattr(self, 'new_round'):
        self.new_round = False
    if hasattr(self, 'after_think'):
        self.after_think = False
    if hasattr(self, 'steering_think_flag'):
        self.steering_think_flag = None

def enable_wait_token_monitoring(self, check_token_ids, punct_token_ids, tokenizer=None):
    """
    Enable wait token monitoring.
    
    Args:
        check_token_ids: List of token IDs to monitor
        punct_token_ids: List of punctuation token IDs to monitor
        tokenizer: Tokenizer for token decoding (optional)
    """
    self.monitor_punct_tokens = True
    self.check_token_ids = check_token_ids
    self.punct_token_ids = punct_token_ids
    self._tokenizer = tokenizer
    logging.info(f"Token monitoring enabled for token IDs: {check_token_ids}, {punct_token_ids}")
   
def cal_sentence_similarity_pair(
    hidden_states,
    sentence_ranges,  # List of (start_idx, end_idx) tuples or List of Lists for multi-batch
    similarity_threshold=0.95,
    aggregation_method="mean"
):
    """
    Remove duplicate sentences based on similarity, keeping the latest occurrence.
    Supports both single batch and multi-batch processing.
    
    Args:
        hidden_states: Hidden states tensor [batch_size, seq_len, hid_dim]
        sentence_ranges: For single batch: List of (start_idx, end_idx) tuples
                        For multi-batch: List[List[(start_idx, end_idx)]] - one list per batch
        similarity_threshold: Threshold above which sentences are considered duplicates
        aggregation_method: How to aggregate tokens within a sentence
    
    Returns:
        For single batch: (sentences_idx, sentence_similarity)
        For multi-batch: List[(sentences_idx, sentence_similarity)] - one tuple per batch
    """
    batch_size = hidden_states.shape[0]
    
    # Determine if we're dealing with multi-batch input
    def process_single_batch(k, batch_sentence_ranges):
        """Process a single batch and return results"""
        seq_len, hid_dim = k.shape
        num_sentences = len(batch_sentence_ranges)
            
        if num_sentences <= 1:
            return None, None
            
        # Aggregate tokens within each sentence
        sentence_representations = []
        valid_sentence_indices = []
        
        for i, (start_idx, end_idx) in enumerate(batch_sentence_ranges):
            if start_idx >= end_idx or end_idx >= seq_len:
                continue
                
            sentence_tokens = k[start_idx:end_idx+1, :]  # [sentence_len, head_dim]

            # Aggregate tokens within sentence
            if aggregation_method == "mean":
                sentence_repr = sentence_tokens.mean(dim=0)  # [head_dim]
            elif aggregation_method == "max":
                sentence_repr = sentence_tokens.max(dim=0)[0]  # [head_dim]
            elif aggregation_method == "last":
                sentence_repr = sentence_tokens[-1, :]  # [head_dim]
            else:
                sentence_repr = sentence_tokens.mean(dim=0)  # Default to mean
            
            sentence_representations.append(sentence_repr)
            valid_sentence_indices.append(i)

        if len(sentence_representations) <= 1:
            return None, None
        
        # Stack sentence representations
        sentence_reprs = torch.stack(sentence_representations, dim=0)  # [num_valid_sentences, head_dim]
        
        # Normalize for cosine similarity
        sentence_reprs_norm = sentence_reprs / (sentence_reprs.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Calculate cosine similarity between sentences
        sentence_similarity = torch.matmul(sentence_reprs_norm, sentence_reprs_norm.transpose(-1, -2))
        # [num_valid_sentences, num_valid_sentences]
        
        # Zero out diagonal and lower triangle (avoid duplicate comparisons)
        sentence_similarity = torch.triu(sentence_similarity, diagonal=1)
        
        # Find sentence pairs above threshold
        similar_pairs = torch.where(sentence_similarity > similarity_threshold)

        range_to_similarity = {}
        if len(similar_pairs[0]) > 0:
            for idx in range(len(similar_pairs[0])):
                i, j = similar_pairs[0][idx].item(), similar_pairs[1][idx].item()
                sim = sentence_similarity[i, j].item()
                # keep later sentence j, mark earlier sentence i
                start_idx, end_idx = batch_sentence_ranges[valid_sentence_indices[i]]
                range_to_similarity[(start_idx, end_idx)] = sim      

        return range_to_similarity, sentence_similarity
    
    # Multi-batch processing
    if sentence_ranges:
        batched_range_to_similarity, batched_sentence_similarity = [], []
        for batch_idx in range(batch_size):
            k = hidden_states[batch_idx]  # [seq_len, head_dim]
            batch_sentence_ranges = sentence_ranges[batch_idx]
            batch_result = process_single_batch(k, batch_sentence_ranges)
            batched_range_to_similarity.append(batch_result[0])
            batched_sentence_similarity.append(batch_result[1])
        return batched_range_to_similarity, batched_sentence_similarity
    else:
        return None, None

def extract_sentence_representations(
        hidden_states, 
        sentence_ranges, 
        sentence_reprs_norm,
        last_token_pos = 0,
        aggregation_method="mean"
        ):
    """
    Extract sentence-level representations from hidden states.
    
    Args:
        hidden_states: Hidden states tensor [batch_size, seq_len, hidden_dim]
        sentence_ranges: List of sentence ranges per batch
        aggregation_method: How to aggregate tokens within a sentence
    
    Returns:
        List of sentence representations per batch
    """
    batch_size = hidden_states.shape[0]

    def process_single_batch(k, last_token_pos, batch_sentence_ranges, batch_sentence_reprs_norm):
        """Process a single batch and return results"""
        seq_len, hid_dim = k.shape     
        
        for i, (start_idx, end_idx) in enumerate(batch_sentence_ranges):

            if start_idx >= end_idx or end_idx >= seq_len or end_idx <= last_token_pos:
                continue
            
            sentence_tokens = k[start_idx:end_idx+1, :]  # [sentence_len, head_dim]
            # sentence_tokens = k[(start_idx - last_token_pos) : (end_idx - last_token_pos + 1), :]  # [sentence_len, head_dim]

            # Aggregate tokens within sentence
            if aggregation_method == "mean":
                sentence_repr = sentence_tokens.mean(dim=0)  # [head_dim]
            elif aggregation_method == "max":
                sentence_repr = sentence_tokens.max(dim=0)[0]  # [head_dim]
            elif aggregation_method == "last":
                sentence_repr = sentence_tokens[-1, :]  # [head_dim]
            else:
                sentence_repr = sentence_tokens.mean(dim=0)  # Default to mean

            sentence_repr_norm = sentence_repr / (sentence_repr.norm(dim=-1, keepdim=True) + 1e-8)
            batch_sentence_reprs_norm[start_idx, end_idx] = sentence_repr_norm


    
    # Multi-batch processing
    if sentence_ranges:
        for batch_idx in range(batch_size):
            k = hidden_states[batch_idx]  # [seq_len, head_dim]
            batch_sentence_ranges = sentence_ranges[batch_idx]
            process_single_batch(k, last_token_pos[batch_idx], batch_sentence_ranges, sentence_reprs_norm[batch_idx])

def cal_sentence_similarity_from_reprs(
    sentence_reprs_norm,
    range_to_similarity,
    last_token_pos=0,
    similarity_threshold=0.95
):
    """
    Faster version of cosine similarity redundancy computation.
    """

    def process_single_batch(sentence_reprs: dict, last_pos: int):
        """Compute redundancy using NEW (start > last_pos) vs ALL sentence representations.

        We compare only newly added sentences against the full set (including themselves)
        and mark EARLIER ranges (smaller start index) as redundant when similarity > threshold.

        Returns:
            range_map: dict mapping earlier redundant ranges -> similarity value
            sim_matrix: Tensor [N_new, N_all] (or None if insufficient)
        """
        if not sentence_reprs or len(sentence_reprs) <= 1:
            return {}, None

        # All existing sentence ranges (in insertion order of dict)
        all_ranges = list(sentence_reprs.keys())
        all_reprs = torch.stack(list(sentence_reprs.values()), dim=0)  # [M, D]

        # Identify new sentences whose start index > last_pos
        new_indices = [i for i, r in enumerate(all_ranges) if r[-1] > last_pos]
        if len(new_indices) == 0:
            return {}, None

        new_ranges = [all_ranges[i] for i in new_indices]
        new_reprs = all_reprs[new_indices]  # [N_new, D]

        # Similarity matrix new @ all (cosine since upstream normalized)
        # sentence_similarity = all_reprs @ all_reprs.T  # [N_new, M]
        # sentence_similarity = torch.triu(sentence_similarity, diagonal=1)
        sentence_similarity = all_reprs @ new_reprs.T  # [N_new, M]
        rows = torch.arange(sentence_similarity.shape[0]).unsqueeze(1).to(sentence_similarity.device)  # shape [N, 1]
        cols = torch.arange(sentence_similarity.shape[1]).unsqueeze(0).to(sentence_similarity.device)  # shape [1, M]
        mask = (rows - cols) < (sentence_similarity.shape[0] - sentence_similarity.shape[1])  # True where we keep upper region
        sentence_similarity = sentence_similarity * mask
        
        # 4. Identify redundant sentences (earlier in each similar pair)
        similar_pairs = torch.where(sentence_similarity > similarity_threshold)
        range_to_similarity = {}

        for i, j in zip(similar_pairs[0].tolist(), similar_pairs[1].tolist()):
            sim = sentence_similarity[i, j].item()
            # earlier_range = all_ranges[i]
            earlier_range = all_ranges[i] if all_ranges[i][0] < new_ranges[j][0] else new_ranges[j]
            range_to_similarity[earlier_range] = sim


        return range_to_similarity, sentence_similarity

    # --- Multi-batch wrapper ---
    if isinstance(sentence_reprs_norm, list):
        sim_mats = []
        for b, batch_reprs in enumerate(sentence_reprs_norm):
            rmap, smat = process_single_batch(batch_reprs, last_token_pos[b])
            range_to_similarity[b].update(rmap)
            sim_mats.append(smat)
        return range_to_similarity, sim_mats
    else:
        rmap, smat = process_single_batch(sentence_reprs_norm, last_token_pos)
        range_to_similarity.update(rmap)
        return range_to_similarity, smat




def find_prev_check(labeled_sentence_ranges):
    """
    Return a list of sentence ranges labeled 'Check' that occurred before the most recent sentence.

    Args:
        labeled_sentence_ranges (list): List of (range_tuple, label) entries.

    Returns:
        List[Tuple[Tuple[int, int], str]]: All previous sentence ranges labeled 'Check'.
    """
    if not labeled_sentence_ranges:
        return []

    sentences_to_remove = []
    for (sent_range, label) in labeled_sentence_ranges:
        if label == "Check":
            sentences_to_remove.append(sent_range)
    if len(sentences_to_remove) > 1:
        return sentences_to_remove[:-1]
    return sentences_to_remove

from transformers.cache_utils import Cache, DynamicCache

def pad_cache(past_key_values: Cache) -> Cache:
    # create a new DynamicCache with padded tensors
    if past_key_values:
        new_cache = DynamicCache()
        for layer_idx, (keys, values) in enumerate(zip(past_key_values.key_cache,
                                            past_key_values.value_cache)):
            batched_len = [k.shape[-2] for k in keys]
            max_batched_len = max(batched_len)
            if isinstance(keys, torch.Tensor):
                new_cache.update(keys, values, layer_idx)
            elif len(set(batched_len)) == 1:
                new_cache.update(torch.stack(keys), torch.stack(values), layer_idx)
            else:
                new_keys = []
                new_values = []
                for b in range(len(keys)):
                    new_k = keys[b]
                    new_v = values[b]
                    if batched_len[b] < max_batched_len:
                        pad_len = max_batched_len - batched_len[b]
                        new_k = torch.nn.functional.pad(new_k, (0, 0, 0, pad_len))  # pad seq_len dim
                        new_v = torch.nn.functional.pad(new_v, (0, 0, 0, pad_len))
                    new_keys.append(new_k)
                    new_values.append(new_v)
                # add into new DynamicCache properly
                new_cache.update(torch.stack(new_keys), torch.stack(new_values), layer_idx)
        return new_cache
    return past_key_values

def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # sample-level statistics
    if len(past_key_values) == 0:
        if self.config.compression_content == "think":
            self.after_think = False

    if not hasattr(self, "length"):
        self.length = input_ids.shape[1]
    else:
        self.length += input_ids.shape[1]

    # Store current input_ids for compression methods to access
    self._current_input_ids = input_ids
    
    ### view removed sentences
    if hasattr(self.config, "method_config"):
        if self.config.method_config['record_kept_token_indices']:
            if not hasattr(self, "generated_token_ids") or len(self.generated_token_ids) != input_ids.shape[0]:
                self.generated_token_ids = [list() for _ in range(input_ids.shape[0])]
            # Append current token (since shape = [batch_size, 1] during decoding)
            for i in range(input_ids.shape[0]):
                self.generated_token_ids[i].extend(input_ids[i].tolist())

    # Find sequences
    if hasattr(self, '_tokenizer'):
        self.check_punct_tokens_in_forward(input_ids, attention_mask)

    # add SEAL steering
    if hasattr(self, "steering_flag"):
        if self.steering_flag:
            if self.new_round:
                self.new_round=False
                # self.steering_think_flag=torch.zeros(input_ids.shape[0], device=input_ids.device).to(torch.bool)
                self.steering_think_flag = (input_ids==self.steering_think_start_id).sum(1).to(torch.bool)
            # else:
                # assert input_ids.shape[1]==1, "use cache"
            last_tokens = input_ids[:,-1]
            self.steering_think_flag = torch.logical_or(self.steering_think_flag, last_tokens==self.steering_think_start_id)
            self.steering_think_flag = torch.logical_and(self.steering_think_flag, last_tokens!=self.steering_think_end_id)
            split_flag = torch.isin(last_tokens, self.steering_split_ids.to(input_ids.device))
            steering_flag = torch.logical_and(split_flag, self.steering_think_flag)
            # logging.info(input_ids, self.steering_think_start_id, self.steering_think_end_id)
            if not torch.any(steering_flag):
                steering_flag = None

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                steering_flag=steering_flag,
                steering_vector=self.steering_vector,
                steering_layer=self.steering_layer,
                steering_coef=self.steering_coef,
            )
        self.cur_steps += 1
    else:
        steering_flag = None


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # past_key_values=pad_cache(past_key_values),
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
    hidden_states = outputs[0] # [batch_size, seq_len, d_hid]

    # Accumulate hidden states over time (optimized with sentence representations)
    if hasattr(self.config, 'method'):
        if self.config.method in ['skipkv']:
            if self._accumulated_hidden_states is None:
                self._accumulated_hidden_states = hidden_states
            else:
                self._accumulated_hidden_states = torch.cat([self._accumulated_hidden_states, hidden_states], dim=1)

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:
        is_newline = False

    # ========== Sentence Similarity Computation (batch 0 only) ==========
    if is_newline and hasattr(self.config, 'method'):
        if self.config.method in ['skipkv']:
            # Handle multi-batch sentence similarity computation
            batch_size = len(self.labeled_sentence_ranges_per_batch)
            if hasattr(self, "completed_punct_ranges") and self.length >= self.config.method_config["budget"]:
                # Multi-batch processing
                last_token_pos = [0] * batch_size
                if any(self._sentence_representations_per_batch):
                    last_token_pos = [list(s.keys())[-1][-1] for s in self._sentence_representations_per_batch]
                extract_sentence_representations(self._accumulated_hidden_states, self.completed_punct_ranges, self._sentence_representations_per_batch, last_token_pos)
                
                cal_sentence_similarity_from_reprs(
                    self._sentence_representations_per_batch, 
                    self._remove_punct_ranges_per_batch,
                    last_token_pos,
                    self.config.method_config['S_threshold'])

                for l, layer in enumerate(self.model.layers): # only use for deep layers
                    layer.self_attn.remove_punct_ranges = self._remove_punct_ranges_per_batch
                # logging.info(f'removed ranges: {batched_remove_punct_ranges}')
            else:
                logging.info("Warning: KV budget not reached, skipping similarity.")

            # Update steering vector
            check_count = torch.zeros(batch_size)
            for i in range(batch_size):
                check_count[i] = len([label for (sent_range, label) in self.labeled_sentence_ranges_per_batch[i] if label == "Check"])
            if hasattr(self, 'steering_coef'):
                if self.steering_coef.dim() == 0:  # scalar tensor
                    self.steering_coef = self.steering_coef.unsqueeze(0).repeat(batch_size)
                # Ensure proper device and dtype for the adjustment tensor
                adjustment = torch.tensor(check_count * self.steering_gamma, device=self.steering_coef.device, dtype=self.steering_coef.dtype)
                self.steering_coef = self.steering_coef_init - adjustment
                logging.info(f'Update steering coef: {self.steering_coef}')

            # ### view removed sentences ###
            # if self.config.method_config['record_kept_token_indices'] and batched_remove_punct_ranges:
            #     # Handle multi-batch debugging
            #     if hasattr(self, "completed_punct_ranges_per_batch") and self.completed_punct_ranges_per_batch:
            #         for batch_idx in range(batch_size):
            #             if batched_remove_punct_ranges[batch_idx]:
            #                 remove_punct_ranges = batched_remove_punct_ranges[batch_idx] if batch_size > 1 else batched_remove_punct_ranges
            #                 sentence_similarity = batched_sentence_similarity[batch_idx] if batch_size > 1 else batched_sentence_similarity
            #                 logging.info(f"Batch {batch_idx} sentence similarity:")
            #                 logging.info(f"Remove punct ranges: {remove_punct_ranges}, Sentence similarity: {sentence_similarity}")
            #                 full_output_ids = self.generated_token_ids[batch_idx]
            #                 debug_text = self._tokenizer.decode(full_output_ids, skip_special_tokens=True)
                            
            #                 remove_ids = [full_output_ids[r[0]:r[1]] for r in remove_punct_ranges]
            #                 for remove_id in remove_ids:
            #                     remove_text = self._tokenizer.decode(remove_id, skip_special_tokens=True)
                                
            #                     # Highlight the removed text within the full text
            #                     debug_text = debug_text.replace(remove_text, f"🚫[TO BE REMOVED: {remove_text}]🚫")
                            
            #                 logging.info(f"\n{'='*80}")
            #                 logging.info(f"Batch {batch_idx} - Final debug text with removed sentences highlighted:")
            #                 logging.info(debug_text)
            #                 logging.info(f"{'='*80}\n")

    # if is_newline and not hasattr(self.config, 'method'): # fullkv + SEAL
    #     batch_size = len(self.labeled_sentence_ranges_per_batch)
    #     check_count = torch.zeros(batch_size)
    #     for i in range(batch_size):
    #         check_count[i] = len([label for (sent_range, label) in self.labeled_sentence_ranges_per_batch[i] if label == "Check"])
    #     if self.steering_coef.dim() == 0:  # scalar tensor
    #         self.steering_coef = self.steering_coef.unsqueeze(0).repeat(batch_size)
    #     # Ensure proper device and dtype for the adjustment tensor
    #     adjustment = torch.tensor(check_count * self.steering_gamma, device=self.steering_coef.device, dtype=self.steering_coef.dtype)
    #     self.steering_coef = self.steering_coef_init - adjustment

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if (
        self.config._attn_implementation == "sdpa"
        and not (using_static_cache or using_sliding_window_cache)
        and not output_attentions
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            sliding_window=self.config.sliding_window,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    # SlidingWindowCache or StaticCache
    if using_sliding_window_cache or using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        config=self.config,
        past_key_values=past_key_values,
    )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask

def Qwen2Model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    steering_flag: Optional[torch.BoolTensor] = None,
    steering_vector: Optional[torch.FloatTensor] = None,
    steering_layer: Optional[int] = None,
    steering_coef: Optional[torch.FloatTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
   
    # # kept for BC (non `Cache` `past_key_values` inputs)
    # return_legacy_cache = False
    # if use_cache and not isinstance(past_key_values, Cache):
    #     return_legacy_cache = True
    #     if past_key_values is None:
    #         past_key_values = DynamicCache()
    #     else:
    #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    #         logger.warning_once(
    #             "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
    #             "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
    #             "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
    #         )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    # next_decoder_cache = None
    for l, decoder_layer in enumerate(self.layers):
        if steering_flag is not None and steering_layer == l:
            # logging.info(f'Steering with {steering_coef}')
            steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
            steering_flag = steering_flag.to(hidden_states.device)
            # Handle batch-wise steering coefficients
            if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
                steering_coef = steering_coef.to(hidden_states.device)
                hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
                    
            else:  # scalar tensor
                hidden_states[steering_flag, -1] += steering_coef * steering_vector # [bsz,seq,dhid], [dhid]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask_mapping[decoder_layer.attention_type],
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
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        # if use_cache:
            # next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if steering_flag is not None and steering_layer == len(self.layers):
        steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
        steering_flag = steering_flag.to(hidden_states.device)
        # Handle batch-wise steering coefficients
        if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
            steering_coef = steering_coef.to(hidden_states.device)
            hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
        else:  # scalar tensor
            hidden_states[steering_flag, -1] += steering_coef * steering_vector

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # next_cache = next_decoder_cache if use_cache else None
    # if return_legacy_cache:
        # next_cache = next_cache.to_legacy_cache()

    # if not return_dict:
        # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def Qwen3Model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    steering_flag: Optional[torch.BoolTensor] = None,
    steering_vector: Optional[torch.FloatTensor] = None,
    steering_layer: Optional[int] = None,
    steering_coef: Optional[torch.FloatTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        )
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
   
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    # next_decoder_cache = None
    for l, decoder_layer in enumerate(self.layers):
        if steering_flag is not None and steering_layer == l:
            # logging.info(f'Steering with {steering_coef}')
            steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
            steering_flag = steering_flag.to(hidden_states.device)
            # Handle batch-wise steering coefficients
            if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
                steering_coef = steering_coef.to(hidden_states.device)
                hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
                    
            else:  # scalar tensor
                hidden_states[steering_flag, -1] += steering_coef * steering_vector # [bsz,seq,dhid], [dhid]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask_mapping[decoder_layer.attention_type],
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
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        # if use_cache:
            # next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if steering_flag is not None and steering_layer == len(self.layers):
        steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
        steering_flag = steering_flag.to(hidden_states.device)
        # Handle batch-wise steering coefficients
        if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
            steering_coef = steering_coef.to(hidden_states.device)
            hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
        else:  # scalar tensor
            hidden_states[steering_flag, -1] += steering_coef * steering_vector

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # next_cache = next_decoder_cache if use_cache else None
    # if return_legacy_cache:
        # next_cache = next_cache.to_legacy_cache()

    # if not return_dict:
        # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def LlamaModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    steering_flag: Optional[torch.BoolTensor] = None,
    steering_vector: Optional[torch.FloatTensor] = None,
    steering_layer: Optional[int] = None,
    steering_coef: Optional[torch.FloatTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # next_decoder_cache = None
    for l, decoder_layer in enumerate(self.layers):
        if steering_flag is not None and steering_layer == l:
            # logging.info(f'Steering with {steering_coef}')
            steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
            steering_flag = steering_flag.to(hidden_states.device)
            # Handle batch-wise steering coefficients
            if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
                steering_coef = steering_coef.to(hidden_states.device)
                hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
                    
            else:  # scalar tensor
                hidden_states[steering_flag, -1] += steering_coef * steering_vector # [bsz,seq,dhid], [dhid]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if steering_flag is not None and steering_layer == len(self.layers):
        steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
        steering_flag = steering_flag.to(hidden_states.device)
        # Handle batch-wise steering coefficients
        if steering_coef.dim() > 0:  # tensor with batch dimension [bsz]
            steering_coef = steering_coef.to(hidden_states.device)
            hidden_states[steering_flag, -1] += steering_coef[steering_flag].unsqueeze(1) * steering_vector.unsqueeze(0)
        else:  # scalar tensor
            hidden_states[steering_flag, -1] += steering_coef * steering_vector

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )