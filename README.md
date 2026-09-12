# Tounges of Gemma

Reproduction and extension of Wendler et al., *Do Llamas Work in English?*, and
RomanLens, on Gemma 3/4 across Hindi, Tamil and Malayalam in native script and
romanisation.

## Layout

```
run.py            sets | corpora | fit-lens | analyse | report | patch
run_all.sh        full sweep: word lists x models x classes x shot counts
build_words.py    RomanLens importer   -> promptutils/words.json
build_controls.py matched en_control   -> promptutils/controls.json
make_slides.py    Beamer decks from the outputs tree
tools_checkfns.py closure-aware unbound-name checker

promptutils/  words.py (the list) + classes.py (six builders)
              words.json, controls.json live here
utils/        models.py vocab.py lenses.py capture.py
plotutils/    plots.py summary.py
patching/     patch.py plots.py

sets_<wordlist>/                       generated prompt JSON
corpora/                               text for Jacobian fitting
lenses/<model>_<corpus>.pt             fitted lenses
outputs/<wordlist>/<model>/<class>_<n>shot/
patchruns/<model>/<pairs>/
```

## Quick start

```bash
python3 build_words.py --romanlens <Romanlens>/llm_logit_lens/data/langs \
    --model ../data/models/gemma-3-4b-pt/
python3 build_controls.py --model ../data/models/gemma-3-4b-pt/
LIMIT=40 bash run_all.sh      # smoke pass, exercises every path
bash run_all.sh
```

## Observables

Notation: `h⁽ˡ⁾` is the residual stream at layer *l* at the final prompt
position, `W_U` the tied unembedding, `RMS` the model's final norm.

**1. Accuracy**, three metrics from greedy decoding: `acc` (substring),
`acc_exact` (first generated word is the answer), `acc_tok1` (top token equals
the answer's first token). `acc_tok1` is the one that matches the curves, since
`P(lang)` is a first-token quantity. Generations are cut at the first closing
quote, because the model continues the few-shot pattern after answering.

**2. P(lang) under any lens.**

```
p⁽ˡ⁾ = softmax( W_U · RMS( transport_l(h⁽ˡ⁾) ) )
P⁽ˡ⁾(ℓ) = Σ_{t ∈ Start(w_ℓ)} p⁽ˡ⁾_t
```

`Start(w)` is every vocabulary token that is a non-blank prefix of *w* or of
`" "+w`. Six labels are tracked per item: `en`, `<lang>_native`,
`<lang>_roman`, a third language, `en_control`, and `src`.

**3. The lens is swappable.** `utils/lenses.py` defines a `Lens` with one
method, `transport(h)`:

| lens | transport | spec on the CLI |
|---|---|---|
| `IdentityLens` | `h` unchanged | `logit` |
| `JacobianLens` | `J_l @ h` | `tag=lenses/x.pt[:lam]` |

Identity returns `h` rather than multiplying by an actual identity matrix —
same semantics, no wasted d×d matmul per layer. A new lens only has to
subclass `Lens`. Several lenses are scored from **one** capture, so adding one
costs no extra forward passes.

Also recorded, because they cost nothing and are needed to read the above:

- **rank** of each label's best token per layer, and **MRR**. Rank is the only
  statistic comparable **across** lenses: transport rescales the residual and
  changes the effective softmax temperature, so `P_en` under J-lens is not on
  the same scale as under logit lens.
- **`L_switch`**, the layer where the target's rank overtakes English, with the
  per-prompt median and spread beside it — a crossover of means is not the mean
  of crossovers.
- **script mass**, `M⁽ˡ⁾(S) = Σ_{t : script(t)=S} p⁽ˡ⁾_t`. Needs no answer key,
  so it works for gibberish, and it is the switch reference for classes with no
  English answer.
- **entropy** per layer, against the `ln(V) ≈ 12.48` uniform ceiling.

## Jacobian lenses

```bash
python3 run.py corpora --out corpora/          # FLORES-200
python3 run.py fit-lens --model M --corpus corpora/en.txt \
    --out lenses/M_en.pt --name jlens_en
python3 run.py analyse --model M --prompts sets_romanlens/*.json \
    --lens logit jlens_en=lenses/M_en.pt jlens_indic=lenses/M_indic.pt
```

`J_l = E[∂h_target/∂h_l]`, computed **exactly**: one backward pass per output
basis direction returns a whole row of J at every source position and layer at
once, so `d_model` passes give the exact Jacobian. Settings follow the Qwen
3.6 27B replication — penultimate target layer, first 4 tokens skipped
(attention sinks), n=25 prompts of 128 tokens.

`fit-lens` ends with a **finite-difference check**. Cosine below 0.3 at mid
layers means the Jacobian is wrong; stop there.

**Corpus choice is the experiment.** FLORES is n-way parallel, so `en.txt` and
`indic.txt` are the *same sentences* — a controlled comparison. No published
J-lens work fits on more than one corpus, so if the mid-stack English survives
only under an English-fit transport, that is a finding.

----------------------------------------------------------------------------------------
## Activation patching 

```bash
python3 run.py patch --model M --out patchruns/M/script --pairs script
```

Reproduces RomanLens Figure 5: two prompts differing only in the **script of
the source word**, both translating into a fixed Latin-script third language
(Italian) so the output side never varies. Patch the residual at each layer and
report **KL between the native-source and romanised-source distributions**.
They report KL < 0.01 on gemma-2-9b-it, concluding the concept is encoded
near-identically regardless of input script.

Worth repeating because their tokenizer fragmented Indic script heavily, while
under Gemma 3 native script costs *fewer* tokens than romanisation. If
invariance survives that, it is a property of the representation rather than of
the tokenization.

`--pairs swap` asks a different question: patch between a consistent prompt
(demos and tag agree) and a conflicting one, to find where the tag overrides
the demonstrations.

## Shot sweep

`run_all.sh` builds classes 1, 2, 3 and 6 at 1, 2, 3 and 4 demonstrations, each
into its own set file and its own output directory. Class 4 is held at 4 shots;
class 5 has no demonstrations.


