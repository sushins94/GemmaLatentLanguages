"""Lenses: a swappable readout from the residual stream to a token distribution.

    p^(l) = softmax( W_U . RMS( transport_l(h^(l)) ) )

Every lens is the same pipeline with a different `transport`:

    IdentityLens   transport = identity          -> the ordinary logit lens
    JacobianLens   transport = J_l @ h           -> J-lens (or R-lens)

IdentityLens returns h unchanged rather than multiplying by an actual identity
matrix -- same semantics, no wasted d x d matmul per layer. A new lens only has
to subclass Lens and implement transport().

WHY TRANSPORT AT ALL
--------------------
Logit lens assumes an activation at layer l already shares coordinates with the
final layer. J-lens drops that: it first maps h into the target layer's basis
using the averaged linearisation of the remaining computation,

    J_l = E[ d h_target / d h_l ]

so the question becomes what the state is disposed to *become* rather than what
it is near.

COMPUTING J EXACTLY
-------------------
A backward pass is reverse-mode: given a scalar it returns the gradient with
respect to every input. Build the scalar with a basis cotangent e_i summed over
target positions, and the backward returns

    d(sum_t h_target[t, i]) / d h_l[s, :]  =  row i of J, at every source
                                              position and every layer at once

so d_model backward passes give the exact Jacobian. That is the O(n * d_model)
cost reported in the Qwen 3.6 27B replication. Settings copied from it: target
the PENULTIMATE layer, skip the first 4 tokens (attention sinks, very high
norm), n = 10-25 prompts of 128 tokens.

CORPUS CHOICE IS AN EXPERIMENT, NOT A SETTING
---------------------------------------------
No published work fits on more than one corpus. If the mid-stack English
survives only under an English-fit transport, you have measured your
calibration text rather than the model. Fit several and compare.
"""

import os

import numpy as np
import torch

from .models import find_backbone


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Lens:
    name = "lens"

    def transport(self, h, device="cuda"):
        """h: [n_layers, d] float32 -> [n_layers, d]."""
        raise NotImplementedError

    @property
    def meta(self):
        return {"name": self.name}


class IdentityLens(Lens):
    """The ordinary logit lens. No transport."""

    name = "logit"

    def transport(self, h, device="cuda"):
        return h


class JacobianLens(Lens):
    """One d x d transport matrix per layer."""

    def __init__(self, J, meta=None, lam=0.0, name=None):
        self.J = J                       # {layer: [d, d] float32 cpu}
        self._meta = meta or {}
        self.lam = lam
        self.name = name or self._meta.get("kind", "jlens")

    def transport(self, h, device="cuda"):
        out = []
        for l in range(h.shape[0]):
            M = self.J[l].to(device)
            if self.lam:
                M = M + self.lam * torch.eye(M.shape[0], device=device,
                                             dtype=M.dtype)
            out.append(h[l] @ M.T)
        return torch.stack(out)

    @property
    def meta(self):
        return {"name": self.name, "lam": self.lam, **self._meta}

    def spectrum(self, layer, k=8):
        s = torch.linalg.svdvals(self.J[layer].float())
        return s[:k].tolist(), float(s.max() / s.median())

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"J": {int(k): v for k, v in self.J.items()},
                    "meta": self._meta}, path)

    @classmethod
    def load(cls, path, lam=0.0, name=None):
        b = torch.load(path, map_location="cpu")
        return cls(b["J"], b.get("meta", {}), lam=lam,
                   name=name or os.path.splitext(os.path.basename(path))[0])


def make_lens(spec):
    """'logit' -> IdentityLens; 'tag=path.pt[:lam]' -> JacobianLens."""
    if spec in ("logit", "identity", "none"):
        return IdentityLens()
    tag, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"lens spec must be 'logit' or 'tag=path.pt', got {spec!r}")
    path, _, lam = rest.partition(":")
    return JacobianLens.load(path, lam=float(lam) if lam else 0.0, name=tag)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

@torch.no_grad()
def lens_probs(h, lens, backbone, W, cap=None, device="cuda"):
    """h: [n_layers, d] -> probabilities [n_layers, vocab].

    float32 throughout regardless of the model's weight dtype: Gemma overflows
    fp16 and bf16 puts ~0.4% noise on every probability, which is the quantity
    being measured. Gemma3RMSNorm upcasts internally and returns the input
    dtype, so feeding it float32 keeps the whole path in float32.
    """
    h = lens.transport(h.to(device=device, dtype=torch.float32), device=device)
    logits = backbone.norm(h) @ W.T
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return torch.softmax(logits.float(), dim=-1)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class _GradCapture:
    def __init__(self, backbone):
        self.backbone, self.tensors, self.handles = backbone, {}, []

    def __enter__(self):
        def make(i):
            def hook(module, args, output):
                self.tensors[i] = (output[0] if isinstance(output, tuple)
                                   else output)
            return hook
        for i, layer in enumerate(self.backbone.layers):
            self.handles.append(layer.register_forward_hook(make(i)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def fit_jacobian(model, tokenizer, texts, n_prompts=25, seq_len=128,
                 skip_first=4, chunk=32, device="cuda", name="jlens"):
    """Exact Jacobians, averaged over prompts and source positions."""
    backbone = find_backbone(model)[1]
    n_layers = len(backbone.layers)
    d = backbone.norm.weight.shape[0]
    target = n_layers - 2                        # penultimate

    acc = {l: torch.zeros(d, d, dtype=torch.float32, device=device)
           for l in range(n_layers)}
    n_pos, used = 0, 0

    for text in texts:
        if used >= n_prompts:
            break
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=seq_len).to(device)
        T = enc["input_ids"].shape[1]
        if T <= skip_first + 4:
            continue

        with _GradCapture(backbone) as cap:
            model(**enc)
            h_tgt = cap.tensors[target]
            tensors = [cap.tensors[l] for l in range(n_layers)]

            for start in range(0, d, chunk):
                idx = torch.arange(start, min(start + chunk, d), device=device)
                C = len(idx)
                cot = torch.zeros(C, 1, T, d, device=device, dtype=h_tgt.dtype)
                cot[torch.arange(C), 0, :, idx] = 1.0
                try:
                    grads = torch.autograd.grad(
                        h_tgt, tensors, grad_outputs=cot, retain_graph=True,
                        allow_unused=True, is_grads_batched=True)
                except Exception:                        # noqa: BLE001
                    grads = [torch.zeros(C, 1, T, d, device=device)
                             for _ in tensors]
                    for c in range(C):
                        g = torch.autograd.grad(
                            h_tgt, tensors, grad_outputs=cot[c],
                            retain_graph=True, allow_unused=True)
                        for l, gl in enumerate(g):
                            if gl is not None:
                                grads[l][c] = gl
                for l, g in enumerate(grads):
                    if g is not None:
                        acc[l][idx] += g[:, 0, skip_first:, :].sum(1).float()

            n_pos += T - skip_first
            used += 1
        model.zero_grad(set_to_none=True)
        print(f"[fit] {used}/{n_prompts} prompts ({T} tokens)")

    if used == 0:
        raise RuntimeError("no usable prompts in the corpus")
    J = {l: (acc[l] / n_pos).cpu() for l in range(n_layers)}
    meta = {"kind": name, "n_prompts": used, "seq_len": seq_len,
            "skip_first": skip_first, "target_layer": target,
            "n_positions": n_pos, "d_model": d, "n_layers": n_layers,
            "method": "exact_basis"}
    print(f"[fit] done: {used} prompts, {n_pos} source positions")
    return JacobianLens(J, meta, name=name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def finite_difference_check(model, lens, tokenizer, layer, eps=1e-2,
                            n_dirs=8, text=None, device="cuda"):
    """Compare J_l @ u against the true directional derivative.

    An averaged Jacobian will not match any single example exactly -- it is a
    corpus average of a nonlinear function -- so read COSINE, not error. Below
    ~0.3 means something is wrong, not merely noisy.

    This validates a J-lens. It does NOT validate an R-lens: LRP deliberately
    departs from the true derivative, so a low cosine there is expected.
    """
    text = text or "The capital city of the country that makes champagne is"
    enc = tokenizer(text, return_tensors="pt").to(device)
    backbone = find_backbone(model)[1]
    n_layers = len(backbone.layers)
    d = backbone.norm.weight.shape[0]
    target = n_layers - 2
    store = {}

    def attach(delta):
        handles = []
        def make(i):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if delta is not None and i == layer:
                    h = h.clone()
                    h[:, -1, :] += delta
                    store[i] = h
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                store[i] = h
                return output
            return hook
        for i, L in enumerate(backbone.layers):
            handles.append(L.register_forward_hook(make(i)))
        return handles

    cos, ratio = [], []
    for _ in range(n_dirs):
        u = torch.randn(d, device=device)
        u = u / u.norm()
        outs = []
        for delta in (None, eps * u.to(next(model.parameters()).dtype)):
            hs = attach(delta)
            model(**enc)
            outs.append(store[target][:, -1, :].detach().float().clone())
            for h in hs:
                h.remove()
        true = (outs[1] - outs[0])[0] / eps
        M = lens.J[layer].to(device)
        pred = u.float() @ M.T
        cos.append(torch.nn.functional.cosine_similarity(
            true, pred.to(true.device), dim=0).item())
        ratio.append((pred.norm() / true.norm().clamp_min(1e-9)).item())

    return {"layer": layer, "cosine": float(np.mean(cos)),
            "cosine_sd": float(np.std(cos)),
            "norm_ratio": float(np.mean(ratio))}