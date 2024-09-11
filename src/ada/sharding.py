
def get_sharding_configs(args):
    num_layers = 32
    
    prefill_sharding = {}
    for layer in range(num_layers):
        prefill_sharding[f"model.layers.{layer}.input_layernorm"] = "data"
        prefill_sharding[f"model.layers.{layer}.self_attn.qkv_proj"] = "data"
        prefill_sharding[f"model.layers.{layer}.self_attn.attn"] = "head"
        prefill_sharding[f"model.layers.{layer}.self_attn.o_proj"] = "data"
        prefill_sharding[f"model.layers.{layer}.post_attention_layernorm"] = "data"
        prefill_sharding[f"model.layers.{layer}.mlp.gate_up_proj"] = "data"
        prefill_sharding[f"model.layers.{layer}.mlp.down_proj"] = "data"
        pass

    decode_sharding = {}

    return prefill_sharding, decode_sharding