import os
import re
import time

import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer, Lfm2MoeForCausalLM


@torch.no_grad()
def port_hf_moe_to_te_grouped(model_opt: Lfm2MoeForCausalLM, repo_or_path: str):
    """
    Map HF per-expert Linear weights (w1, w3, w2) into TransformerEngine GroupedLinear
    in an optimized Lfm2Moe* model.

    Works for both TE layouts:
      - Layout A: a single 3-D tensor gl.weight shaped [E, out, in]
      - Layout B: per-expert parameters gl.weight0, gl.weight1, ... (your case)
      - (Optional) 'weights' ParameterList fallback

    Assumes bias=False in both paths (zeros bias if present).
    """

    # 1) Load a non-optimized reference checkpoint
    cfg = AutoConfig.from_pretrained(repo_or_path)
    cfg.moe_use_optimized = False
    ref = Lfm2MoeForCausalLM.from_pretrained(
        repo_or_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, config=cfg
    )
    sd = ref.state_dict()
    del ref

    def _find_expert_param(gl, e):
        """
        Return a writable tensor for expert e in GroupedLinear gl.
        Supports:
          - gl.weight (E, out, in)
          - gl.weight{e} attributes
          - gl.weights ParameterList
        """
        # Layout A: a single stacked tensor
        if hasattr(gl, "weight") and isinstance(gl.weight, torch.Tensor) and gl.weight.ndim == 3:
            if e < gl.weight.shape[0]:
                return gl.weight[e]
            raise IndexError(f"GroupedLinear.weight has {gl.weight.shape[0]} groups, want {e}")

        # Layout B: per-expert attributes weight0, weight1, ...
        attr_name = f"weight{e}"
        if hasattr(gl, attr_name):
            return getattr(gl, attr_name)

        # Optional fallback: ParameterList named 'weights'
        if hasattr(gl, "weights"):
            try:
                return gl.weights[e]
            except Exception as ex:
                raise RuntimeError(f"'weights' list exists but index {e} failed: {ex}")

        # As a last resort, search named_parameters for weight{e}
        for name, p in gl.named_parameters(recurse=False):
            if name == attr_name:
                return p

        # Nothing matched
        avail = [n for n, _ in gl.named_parameters(recurse=False)]
        raise AttributeError(f"Cannot locate expert slice {attr_name} in GroupedLinear; available: {avail}")

    def _copy_into(dst: torch.Tensor, src: torch.Tensor, tag: str):
        if src.shape != dst.shape:
            raise ValueError(f"[port] shape mismatch for {tag}: ckpt {tuple(src.shape)} vs dst {tuple(dst.shape)}")
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))

    copied = 0
    missing = 0
    num_layers = model_opt.config.num_hidden_layers

    for L in range(num_layers):
        layer = model_opt.model.layers[L]
        ff = getattr(layer, "feed_forward", None)
        if ff is None:
            continue

        experts = getattr(ff, "experts", None)
        if experts is None or not getattr(experts, "moe_use_optimized", False):
            # Not an optimized MoE layer (i.e., dense FFN or slow-path experts)
            continue

        # TE modules in your code are named like this:
        gl_up   = getattr(experts, "_fc_up",   None)
        gl_gate = getattr(experts, "_fc_gate", None)
        gl_down = getattr(experts, "_fc_down", None)
        if gl_up is None or gl_gate is None or gl_down is None:
            print(f"[port][WARN] L{L}: missing grouped linears; skipping.")
            continue

        E = experts.num_experts

        for e in range(E):
            base = f"model.layers.{L}.feed_forward.experts.{e}"
            k_w1 = f"{base}.w1.weight"  # [ffn_hidden, hidden_size]
            k_w3 = f"{base}.w3.weight"  # [ffn_hidden, hidden_size]
            k_w2 = f"{base}.w2.weight"  # [hidden_size,  ffn_hidden]

            if (k_w1 not in sd) or (k_w3 not in sd) or (k_w2 not in sd):
                # If ckpt doesn’t have these, keep counting but move on
                missing += 1
                # Print once per layer to avoid spam
                if e == 0:
                    print(f"[port][WARN] L{L}: missing keys like {k_w1}/{k_w3}/{k_w2}")
                continue

            # Locate per-expert destination tensors (handles both layouts)
            dst_up   = _find_expert_param(gl_up,   e)   # shape [ffn_hidden, hidden_size]
            dst_gate = _find_expert_param(gl_gate, e)   # shape [ffn_hidden, hidden_size]
            dst_down = _find_expert_param(gl_down, e)   # shape [hidden_size, ffn_hidden]

            # Copy directly; no transpose needed
            _copy_into(dst_up,   sd[k_w1], f"L{L}.E{e}.up")
            _copy_into(dst_gate, sd[k_w3], f"L{L}.E{e}.gate")
            _copy_into(dst_down, sd[k_w2], f"L{L}.E{e}.down")
            copied += 3

        # If bias exists, keep it zeroed for parity with HF path
        for gl in (gl_up, gl_gate, gl_down):
            if getattr(gl, "bias", None) is not None:
                gl.bias.zero_()

    print(f"[port] copied tensors: {copied} | experts missing in ckpt: {missing}")

    # Quick nonzero smoke test on the first optimized MoE layer we touched
    for L in range(num_layers):
        layer = model_opt.model.layers[L]
        ff = getattr(layer, "feed_forward", None)
        experts = getattr(ff, "experts", None)
        if experts is None or not getattr(experts, "moe_use_optimized", False):
            continue
        gl_up = experts._fc_up
        gl_gate = experts._fc_gate
        gl_down = experts._fc_down

        def _nz(gl):
            # Support both layouts
            if hasattr(gl, "weight") and isinstance(gl.weight, torch.Tensor) and gl.weight.ndim == 3:
                return int((gl.weight != 0).sum().item())
            total = 0
            for name, p in gl.named_parameters(recurse=False):
                if re.fullmatch(r"weight\d+", name):
                    total += int((p != 0).sum().item())
            if total == 0 and hasattr(gl, "weights"):
                for p in gl.weights:
                    total += int((p != 0).sum().item())
            return total

        print(f"[port] L{L} nonzeros — up:{_nz(gl_up)} gate:{_nz(gl_gate)} down:{_nz(gl_down)}")
        break


def init_dist():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",   # torchrun sets RANK, WORLD_SIZE, MASTER_* envs
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    DEBUG = False
    local_rank = init_dist()

    config = AutoConfig.from_pretrained("LiquidAI/LFM2-8B-A1B")
    config.moe_use_optimized = True
    config.moe_dropless = True
    config.moe_capacity_factor = 1.25
    config.moe_debug = DEBUG
    config.moe_fp8_enable = False

    model = Lfm2MoeForCausalLM.from_pretrained("/lambdafs/anna/lfm2-8b-a1b-te-grouped", torch_dtype=torch.bfloat16, device_map=local_rank, config=config)
    # port_hf_moe_to_te_grouped(model, "LiquidAI/LFM2-8B-A1B")
    # model.save_pretrained("/lambdafs/anna/lfm2-8b-a1b-te-grouped")

    config_ref = AutoConfig.from_pretrained("LiquidAI/LFM2-8B-A1B")
    config_ref.moe_debug = DEBUG

    ref_model = Lfm2MoeForCausalLM.from_pretrained("LiquidAI/LFM2-8B-A1B", torch_dtype=torch.bfloat16, device_map=local_rank, config=config_ref)
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-8B-A1B")

    # Generate answer
    prompt = "What is the capital of Poland?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(local_rank)
    attempts = 10
    start_time = time.time()
    for _ in range(attempts):
        output_opt = model.generate(
            input_ids,
                do_sample=False,
                max_new_tokens=256,
            )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Time per attempt: {(end_time - start_time) / attempts} seconds")

    start_time = time.time()
    for _ in range(attempts):
        output_ref = ref_model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=256,
        )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Time per attempt: {(end_time - start_time) / attempts} seconds")

    print(f"Optimized: {tokenizer.decode(output_opt[0], skip_special_tokens=True)} \n")
    print(f"Reference: {tokenizer.decode(output_ref[0], skip_special_tokens=True)}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()