import os
import argparse

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer

from accelerate import init_empty_weights

def setup_gpt2_models(args):
    HEAD_SIZE = 64
    save_path = args.save_path
    tokenizer = AutoTokenizer.from_pretrained(f"openai-community/gpt2-xl")

    checkpoints = [
        "nice-gpt2-1.5b",
        "nice-gpt2-4b",
        "nice-gpt2-5.7b",
        "nice-gpt2-11.2b",
        "nice-gpt2-30.1b",
        "nice-gpt2-68.8b",
    ]
    configs = []

    num_head_list = [24, 40, 48]
    for num_heads in num_head_list:
        config = GPT2Config(
            n_layer=48,
            n_head=num_heads,
            n_embd=num_heads * HEAD_SIZE,
            n_positions = 16384,
        )
        config.activation_function = "gelu"
        configs.append(config)

    num_head_list = [48, 80, 120]
    for num_heads in num_head_list:
        config = GPT2Config(
            n_layer=96,
            n_head=num_heads,
            n_embd=num_heads * HEAD_SIZE,
            n_positions = 16384,
        )
        config.activation_function = "gelu"
        configs.append(config)


    for idx, config in enumerate(configs):
        print(f"Checkpoint: {checkpoints[idx]}")
        print(
            f"num heads: {config.n_head}, num layers: {config.n_layer}, emb size: {config.n_embd}"
        )

        # Uncomment below to initialize quickly
        # with init_empty_weights():
        model = GPT2LMHeadModel(config)

        # # Get parameter count
        params = sum(p.numel() for p in model.parameters())
        print(f"params: {params:,}")

        model.save_pretrained(f"{save_path}/{checkpoints[idx]}")
        tokenizer.save_pretrained(f"{save_path}/{checkpoints[idx]}")

def setup_llama_models(args):
    save_path = args.save_path
    
    checkpoints = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-chat-hf",
    ]

    for checkpoint in checkpoints:
        print(f"Checkpoint: {checkpoint}")
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.save_pretrained(f"{save_path}/{checkpoint}")
        print(f"Loading model...")
        config = LlamaConfig.from_pretrained(checkpoint)
        config.max_position_embeddings = 16384
        model = LlamaForCausalLM(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"params: {params:,}")
        model.save_pretrained(f"{save_path}/{checkpoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Action (running mode)
    # --------------------------------------------
    action_group = parser.add_argument_group("Action (running mode)")
    action_group.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="Run autosharding on a single design point, defined by batch size and sequence length.",
    )

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.environ["ADASERVE_CHECKPOINTS_PATH"]

    setup_gpt2_models(args)
    setup_llama_models(args)