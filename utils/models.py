"""Model loading and structure discovery for Gemma."""

import torch


def load_model(path, device="cuda", dtype=None):
    """Try loader classes in order of specificity.

    Gemma 3 4B/12B are multimodal, so they load as
    Gemma3ForConditionalGeneration with the decoder nested one level down.
    Gemma 4 is encoder-free and may expose a flatter tree. Nothing here
    hardcodes a path.

    NEVER float16: Gemma's final-norm weights reach ~53 and the residual stream
    overflows fp16's 65504 ceiling, giving NaN. float32 for analysis, bfloat16
    for 12B-class weights and for Jacobian fitting.
    """
    import transformers

    dtype = dtype or torch.float32
    if dtype == torch.float16:
        raise SystemExit("float16 overflows on Gemma; use float32 or bfloat16")
    last = None
    for name in ("Gemma3ForConditionalGeneration",
                 "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(path, torch_dtype=dtype,
                                    attn_implementation="eager",
                                    device_map=device)
            print(f"[model] {name} -> {type(m).__name__} on {device} "
                  f"({str(dtype).replace('torch.', '')})")
            return m
        except Exception as e:                          # noqa: BLE001
            last = e
            print(f"[model] {name} failed: {type(e).__name__}")
    raise RuntimeError(f"all loaders failed; last error: {last}")


def find_backbone(model):
    """-> (dotted path, module, n_layers) for the decoder stack."""
    best = None
    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm"):
            try:
                n = len(mod.layers)
            except TypeError:
                continue
            if n > 1 and (best is None or n > best[2]):
                best = (name, mod, n)
    if best is None:
        raise RuntimeError("could not find the decoder stack")
    return best


def find_softcap(model):
    """Gemma 2 applied final logit softcapping; Gemma 3 dropped it. Read the
    config rather than assuming -- getting this wrong silently distorts every
    probability."""
    for cfg in (getattr(model.config, "text_config", None), model.config):
        if cfg is None:
            continue
        c = getattr(cfg, "final_logit_softcapping", None)
        if c:
            return float(c)
    return None


def unembed_fp32(model):
    W = model.get_output_embeddings().weight
    return W if W.dtype == torch.float32 else W.float()


def make_lm(model, tokenizer, backbone_path):
    """nnsight wrapper plus the envoy for the decoder stack."""
    from nnsight import LanguageModel
    lm = LanguageModel(model, tokenizer=tokenizer)
    envoy = lm
    for part in backbone_path.split("."):
        envoy = envoy[int(part)] if part.isdigit() else getattr(envoy, part)
    return lm, envoy


def describe(model, tokenizer):
    path, backbone, n_layers = find_backbone(model)
    W = unembed_fp32(model)
    info = {"class": type(model).__name__, "backbone_path": path,
            "n_layers": n_layers,
            "d_model": int(backbone.norm.weight.shape[0]),
            "vocab_head": int(W.shape[0]),
            "vocab_tokenizer": len(tokenizer),
            "softcap": find_softcap(model),
            "tied": bool(W.data_ptr()
                         == model.get_input_embeddings().weight.data_ptr())}
    for k, v in info.items():
        print(f"[model] {k:<16} {v}")
    assert info["vocab_head"] >= info["vocab_tokenizer"], \
        "unembedding smaller than tokenizer vocab; ids would overflow"
    return info