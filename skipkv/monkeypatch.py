from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from .modeling_nhid import (
    LlamaAttention_init,
    LlamaAttention_forward,
    Qwen2Attention_init,
    Qwen2Attention_forward,
    Qwen3Attention_init,
    Qwen3Attention_forward,
    CausalLM_forward,
    CausalLM_init,
    set_steering_flag,
    start_new_round,
    Qwen2Model_forward,
    Qwen3Model_forward,
    LlamaModel_forward,
    Qwen2ForCausalLM_init,
    Qwen3ForCausalLM_init,
    LlamaForCausalLM_init,
    _update_causal_mask,
    enable_wait_token_monitoring,
    check_punct_tokens_in_forward,
    reset_for_new_sample
)


def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    modeling_llama.LlamaForCausalLM.__init__ = LlamaForCausalLM_init
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward

    modeling_llama.LlamaForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_llama.LlamaForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_llama.LlamaForCausalLM.reset_for_new_sample = reset_for_new_sample

def replace_qwen2(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen2Attention_init(self, config, layer_idx, compression_config)
        
    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward
    modeling_qwen2.Qwen2ForCausalLM.__init__ = Qwen2ForCausalLM_init
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward
    
    modeling_qwen2.Qwen2ForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_qwen2.Qwen2ForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_qwen2.Qwen2ForCausalLM.reset_for_new_sample = reset_for_new_sample

def replace_llama_steering():
    # modeling_llama.LlamaAttention.__init__ = init_wrapper
    # modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    
    modeling_llama.LlamaModel.forward = LlamaModel_forward
    modeling_llama.LlamaModel._update_causal_mask = _update_causal_mask

    modeling_llama.LlamaForCausalLM.__init__ = CausalLM_init
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward
    modeling_llama.LlamaForCausalLM.set_steering_flag = set_steering_flag
    modeling_llama.LlamaForCausalLM.start_new_round = start_new_round

    modeling_llama.LlamaForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_llama.LlamaForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_llama.LlamaForCausalLM.reset_for_new_sample = reset_for_new_sample

def replace_qwen2_steering():
    # modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    # modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward

    modeling_qwen2.Qwen2Model.forward = Qwen2Model_forward
    modeling_qwen2.Qwen2Model._update_causal_mask = _update_causal_mask
    
    modeling_qwen2.Qwen2ForCausalLM.__init__ = CausalLM_init
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward # fullkv + SEAL...
    modeling_qwen2.Qwen2ForCausalLM.set_steering_flag = set_steering_flag
    modeling_qwen2.Qwen2ForCausalLM.start_new_round = start_new_round

    modeling_qwen2.Qwen2ForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_qwen2.Qwen2ForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_qwen2.Qwen2ForCausalLM.reset_for_new_sample = reset_for_new_sample

def replace_qwen3_steering():
    # modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    # modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward

    modeling_qwen3.Qwen3Model.forward = Qwen3Model_forward
    modeling_qwen3.Qwen3Model._update_causal_mask = _update_causal_mask
    
    modeling_qwen3.Qwen3ForCausalLM.__init__ = CausalLM_init
    modeling_qwen3.Qwen3ForCausalLM.forward = CausalLM_forward # fullkv + SEAL...
    modeling_qwen3.Qwen3ForCausalLM.set_steering_flag = set_steering_flag
    modeling_qwen3.Qwen3ForCausalLM.start_new_round = start_new_round

    modeling_qwen3.Qwen3ForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_qwen3.Qwen3ForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_qwen3.Qwen3ForCausalLM.reset_for_new_sample = reset_for_new_sample

def replace_qwen3(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3.Qwen3Attention.__init__ = init_wrapper
    modeling_qwen3.Qwen3Attention.forward = Qwen3Attention_forward
    modeling_qwen3.Qwen3ForCausalLM.__init__ = Qwen3ForCausalLM_init
    modeling_qwen3.Qwen3ForCausalLM.forward = CausalLM_forward
    
    modeling_qwen3.Qwen3ForCausalLM.enable_wait_token_monitoring = enable_wait_token_monitoring
    modeling_qwen3.Qwen3ForCausalLM.check_punct_tokens_in_forward = check_punct_tokens_in_forward
    modeling_qwen3.Qwen3ForCausalLM.reset_for_new_sample = reset_for_new_sample
