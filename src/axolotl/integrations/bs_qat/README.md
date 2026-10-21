# BS_QAT plugin

Axolotl integration that fake-quantizes selected `nn.Linear` layers to
ternary/binary BitLinear weights during training, for export to a baked
Q2_0 GGUF (see `prism-ml/Ternary-Bonsai-4B-unpacked` and the
Bonsai 1-bit LLM framework).

## Background: what an Axolotl plugin is

An Axolotl plugin (`axolotl.integrations.base.BasePlugin`) is a class of
lifecycle hooks that Axolotl calls at fixed points around model
construction, adapter attachment, and training — the same mechanism used by
every integration under `src/axolotl/integrations/` (`liger`,
`cut_cross_entropy`, `llama_cpp_qat`, this plugin, etc.). Nothing here is
Plugin-specific; it's the general contract, restated because the three LoRA
modes above only make sense once the hook ordering is clear.

Hooks fire in roughly this order during a training run (see
`src/axolotl/integrations/base.py` for the authoritative list/signatures):

1. `register(cfg)` — plugin is loaded from the `plugins:` config list.
2. `get_input_args()` — returns the dotted path to a Pydantic model
   (`args.py` here) that Axolotl merges into the config schema, so your
   plugin's YAML keys get validated like any built-in Axolotl option.
3. `pre_model_load(cfg)` — before the HF model is constructed at all.
4. `post_model_build(cfg, model)` — model exists, **no PEFT adapter yet**.
5. `pre_lora_load(cfg, model)` / `post_lora_load(cfg, model)` — around
   Axolotl attaching a PEFT adapter (only relevant if `adapter: lora/qlora`
   is set; skipped entirely otherwise).
6. `post_model_load(cfg, model)` — model is fully assembled, adapter
   included if any. Runs regardless of adapter mode, so it's a common place
   for idempotent "make sure X happened" fallback logic.
7. `get_trainer_cls`, `get_training_args`, `get_collator_cls_and_kwargs`,
   `create_optimizer`, `create_lr_scheduler` — let a plugin swap out
   Trainer-construction pieces instead of just touching the model.
8. `post_trainer_create(cfg, trainer)`, `add_callbacks_pre_trainer` /
   `add_callbacks_post_trainer` — register `TrainerCallback`s once the
   `Trainer` object exists (this plugin uses the post-trainer hook for
   `InternalQATLoRASaveCallback`, see below).
9. `post_train(cfg, model)` / `post_train_unload(cfg)` — after training
   finishes, before/after the model is torn down.

Other opt-in hooks not used by this plugin: `load_datasets` (custom dataset
loading), `get_adapter_capabilities`/`get_lora_config_kwargs`/`load_adapter`
(to contribute a new adapter *type*, distinct from just modifying an existing
one), `on_rollouts_scored` (RL/GRPO reward post-processing).

### Writing a new plugin

1. Create `src/axolotl/integrations/<name>/{__init__.py,plugin.py,args.py}`.
2. `args.py`: a Pydantic `BaseModel` (see `args.py` in this directory) with
   your config fields; `get_input_args()` on your plugin class returns its
   dotted import path so Axolotl's schema validation picks it up.
3. `plugin.py`: subclass `BasePlugin`, override only the hooks you need —
   every hook has a no-op default, so a minimal plugin can implement just
   one (e.g. `post_model_build` to patch modules, or
   `add_callbacks_post_trainer` to add a `TrainerCallback`).
4. Reference it from a config's `plugins:` list by dotted path
   (`axolotl.integrations.<name>.<PluginClass>`), same as this plugin's
   `axolotl.integrations.bs_qat.BSQATPlugin`.
5. Guard config combinations that don't make sense together by raising in an
   early hook (`_validate_internal_qat_cfg`, called from `post_model_build`
   here, is the pattern to copy — see the `internal_qat`/`adapter:` guard
   below) rather than letting a later hook fail confusingly.

The rest of this README documents how `BSQATPlugin` uses these hooks
specifically.

## Config

```yaml
plugins:
  - axolotl.integrations.bs_qat.BSQATPlugin

bs_qat: true
bs_qat_mode: "ternary"          # or "binary"
bs_qat_group_size: 128
bs_qat_act_quant: false
bs_qat_target_names: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
bs_qat_skip_names: [lm_head]    # never fake-quantized, trained or exported
```

## LoRA/QAT composition modes

`bs_qat_lora_mode` selects how LoRA and fake-quant compose:

| | `external_peft` (default) | `internal_qat` | `full_weight` |
|---|---|---|---|
| LoRA owner | Axolotl/PEFT adapter, outside BitLinear | `BitLinear.lora_A`/`lora_B`, inside the layer | none — no LoRA anywhere |
| Forward path | `QDQ(W_base) + LoRA_delta` | `QDQ(W_base + LoRA_delta)` | `QDQ(W)`, `W` directly trainable |
| Axolotl `adapter:` | `lora` (PEFT attaches normally) | must be unset — PEFT must never wrap the model | must be unset — same reason |
| Hook that patches BitLinear | `post_lora_load` (after PEFT wraps LoRA) | `post_model_build` (before PEFT would attach) | `post_model_load` (no PEFT hook ever fires when `adapter:` is unset) |
| Checkpoint | standard PEFT `adapter_model.safetensors` | Trainer full-model checkpoint + lightweight sidecar `checkpoint-<step>/internal_qat_lora/bs_internal_qat_lora.safetensors` (`bitnet_qat/internal_lora.py`) | Trainer full-model checkpoint only — no sidecar, since there is no separate LoRA state to extract |
| Export merge step | `model.merge_and_unload()` before baking | no PEFT merge — LoRA is already inside the baked weight | no merge — the checkpoint's weights bake directly |

`internal_qat` matches the training-time math to the deployed baked GGUF
objective (`QDQ(W + LoRA)`); `external_peft` is the older path where a single
Q2_0 bake of `W_base + LoRA_delta` can diverge from what was actually trained
(`QDQ(W_base) + LoRA_delta`). `full_weight` is full-parameter QAT: every target
`BitLinear.weight` is trainable and fake-quantized directly, with no LoRA
delta at all. Code-path-wise it patches identically to `external_peft`
(`BitLinear(use_lora=False)`) — the separate name exists so a config that
intentionally has no adapter doesn't have to claim `external_peft`, which
implies a PEFT adapter exists outside the layer (it doesn't, in this mode).

### Exact forward formulas

`external_peft` — `BitLinear` has no LoRA of its own (`use_lora=False`); the
quantized weight is cached and PEFT's adapter runs as an outer wrapper:

```
W_eff = FakeQuant(W)                              # cached, only changes if W changes
y     = x @ W_effᵀ + bias  +  (α/r) · x @ (B·Dropout(A))ᵀ
```

`internal_qat` — `BitLinear.forward` (`bitnet_qat/bitlinear.py`) adds the LoRA
delta to the shadow weight *before* fake-quantizing, every forward pass (not
cacheable since the effective weight changes with `A`/`B`):

```
W_eff = FakeQuant( W + (α/r) · B · Dropout(A) )
y     = x @ W_effᵀ + bias
```

`FakeQuant` is a group-wise (size `bs_qat_group_size`) round-to-ternary-
grid-then-dequantize op with a straight-through gradient estimator, so
gradients flow to `W` (frozen in `internal_qat`, since only `lora_A`/`lora_B`
are trainable) and to `A`/`B` as if the rounding were the identity.

The practical difference: `external_peft` quantizes `W` alone and adds the
LoRA delta in full precision afterward; `internal_qat` quantizes `W + LoRA`
as one tensor, exactly like the single-tensor baked Q2_0 export has to.

Internal-QAT LoRA hyperparameters (`bs_qat_lora_r/alpha/dropout`) fall
back to the corresponding top-level `lora_r/lora_alpha/lora_dropout` keys when
unset, so an existing external_peft config can switch modes with minimal
changes to hyperparameters.

## Provenance: `internal_qat` is the original Bonsai design

`internal_qat` is not a new training method invented for this integration —
it is a direct port of how the standalone
Bonsai codebase
(`bs_bonsai/src/bitnet_qat/`) already fine-tunes, before any Axolotl
involvement. Its `BitLinear` has always carried `lora_A`/`lora_B` as instance
attributes and added the LoRA delta into the shadow weight *before*
fake-quantizing in `forward()`; the standalone trainer (`bitnet_qat/cli.py`)
patches `nn.Linear -> BitLinear(use_lora=True, ...)` directly and never
imports `peft`. `external_peft` is the mode that is new here — it exists
because Axolotl's training loop, checkpointing, and eval assume a PEFT
adapter is attached, so getting the plugin running inside Axolotl at all was
easiest by keeping BitLinear PEFT-agnostic (no LoRA of its own) and letting
Axolotl's normal `adapter: lora` path wrap it, same as any other backbone.
`internal_qat` closes that gap: it reuses BitLinear's original LoRA fields but
drives them from inside an Axolotl plugin hook instead of the standalone
`bitnet_qat` trainer.

### How the port into Axolotl works

Axolotl plugins (`axolotl.integrations.base.BasePlugin`) expose lifecycle
hooks that fire around model construction; the constraint driving this
integration is that PEFT normally attaches its adapter modules in the middle
of that lifecycle, and `internal_qat` needs `BitLinear` — with its LoRA
already inside it — in place *before* that happens, so PEFT has nothing left
to wrap:

- `post_model_build(cfg, model)` — fires right after the HF model is
  constructed, before Axolotl would call into PEFT. `BSQATPlugin` uses
  this hook (only) for `internal_qat`: it patches the target `nn.Linear`
  modules straight to `BitLinear(use_lora=True, lora_r=..., ...)`
  (`_replace_target_linears` in `plugin.py`) and immediately freezes every
  parameter except `lora_A`/`lora_B` (`mark_only_internal_lora_as_trainable`).
- `_validate_lora_mode_cfg` raises at config-load time if `adapter:` is set
  alongside `bs_qat_lora_mode: internal_qat` (or `full_weight`) — this
  is the guard that keeps PEFT out entirely, since Axolotl would otherwise
  attach an adapter on top of an already-fully-trainable `BitLinear` (either
  because it already has its own LoRA, for `internal_qat`, or because there is
  nothing left for a PEFT delta to usefully do, for `full_weight`). It runs
  from inside `_apply_qat`, so it fires no matter which hook ends up doing the
  actual patch for a given mode.
- `post_lora_load`/`post_model_load` are no-ops for `internal_qat` (they only
  run the `external_peft`/`full_weight` patch path); `_apply_qat` is
  idempotent either way (checks for existing `BitLinear` instances before
  patching) so the hooks can safely all fire on the same load without
  re-wrapping. `full_weight` patches at `post_model_load` specifically,
  because `post_lora_load` never fires at all when `adapter:` is unset.
- Because `internal_qat` never has a PEFT adapter, Axolotl's default
  Trainer checkpointing saves a full model rather than an adapter dir. The
  integration adds `InternalQATLoRASaveCallback` (a `TrainerCallback`
  registered via `add_callbacks_post_trainer`) purely to keep checkpoints
  cheap to consume downstream: on every `on_save` it writes a lightweight
  `lora_A`/`lora_B`-only sidecar next to the Trainer checkpoint
  (`bitnet_qat/internal_lora.py::save_internal_lora`), and `post_train` writes
  one more such sidecar into `output_dir` when training ends. Neither
  sidecar replaces the full Trainer checkpoint; they only make it cheap to
  resume PPL eval / export without deserializing the whole model. This
  callback is registered only for `internal_qat` — `full_weight` has no LoRA
  tensors to extract into a sidecar, so its checkpoints are plain Trainer
  full-model saves with nothing extra alongside them.

## lm_head / embeddings

`lm_head` is always in `bs_qat_skip_names` and is never fake-quantized
during training, at either LoRA mode. Token embeddings are never touched by
this plugin (only `nn.Linear` targets are patched).

This matters at export time: the original `bitnet_qat.export.requantize_gguf`
GGUF repacker (from the upstream Bonsai framework) quantizes `output.weight`
(`lm_head`) to Q2_0 by default and only skips embeddings if
`quantize_embeddings=False` is left at its default. Applying that default to
this integration's exports would post-hoc quantize a tensor that was never
QAT-trained, which can silently degrade quality. Use
`bitnet_qat/export/requantize_gguf_v2.py` instead — it defaults to the seven
QAT-target projection tensors used by this config and leaves `output.weight`
and `token_embd.weight` at full precision unless explicitly requested.

## Export pipeline

Three scripts, run in order. Every GGUF produced by this integration goes
through all three — there is no path that skips straight to Q2_0.

1. **`scripts/export_bs_qat_lora_baked_hf.py`** — load base model,
   patch `BitLinear`, then per `--lora-mode`:
   - `external_peft`: `PeftModel.from_pretrained(--checkpoint)`, patch, then
     `model.merge_and_unload()` before baking.
   - `internal_qat`: patch with `use_lora=True`, then load the
     `--checkpoint`'s internal-QAT sidecar or full Trainer state
     (`load_full_or_internal_lora_state`) into the already-patched model —
     no PEFT merge.
   - `full_weight`: patch with `use_lora=False` (no LoRA slots to create),
     then load the `--checkpoint`'s full Trainer state directly
     (`load_full_or_internal_lora_state`, which falls back to
     `load_full_checkpoint_state_dict` since a full_weight checkpoint never
     has the internal-QAT sidecar) — also no merge, since there's no LoRA
     delta to fold in.
   - `none`: no `--checkpoint` needed, patch and bake the base model as-is
     (added specifically for base-vs-finetuned GGUF comparisons, e.g. the
     PPL sanity check below).
   All four modes end by baking `BitLinear` to plain `nn.Linear` with
   on-grid weights (`bake_bitlinear_to_linear`) and calling
   `save_pretrained` on a normal HF directory.
2. **`convert_hf_to_gguf.py`** (from the Prism `llama.cpp` fork checked out
   at `../llama.cpp`, *not* this repo) — baked HF directory → F16 GGUF. This
   step needs the fork specifically because it knows the Q2_0-aware GGUF
   metadata/tensor naming this integration's Q2_0 packer expects.
3. **`bitnet_qat/export/requantize_gguf_v2.py`** — F16 GGUF → Q2_0 GGUF,
   repacking only tensors that were actually fake-quant trained by default
   (see the lm_head/embeddings section above). Pass `--include-output
   --include-embeddings` to force-quantize those too (e.g. to match a
   specific deployment target) — note the base model here ties
   `lm_head`/embedding weights, so in practice `--include-output` is a
   no-op and `--include-embeddings` covers both (GGUF only ever wrote one
   `token_embd.weight` tensor to begin with; there's no separate
   `output.weight` to find).

Example, mode `none` (bake the fake-quantized base model, no LoRA, for a
baseline GGUF):

```bash
python scripts/export_bs_qat_lora_baked_hf.py \
  --lora-mode none \
  --output-dir runs/q2_0_base_gguf_compare/baked_base_hf

python ../llama.cpp/convert_hf_to_gguf.py \
  runs/q2_0_base_gguf_compare/baked_base_hf \
  --outtype f16 \
  --outfile runs/q2_0_base_gguf_compare/base-f16.gguf

python -m axolotl.integrations.bs_qat.bitnet_qat.export.requantize_gguf_v2 \
  runs/q2_0_base_gguf_compare/base-f16.gguf \
  runs/q2_0_base_gguf_compare/base-Q2_0-full.gguf \
  --include-output --include-embeddings \
  --manifest runs/q2_0_base_gguf_compare/requantize_manifest.json
```

`scripts/eval_qat_lora_ppl.py` computes perplexity against a checkpoint or
baked export for regression checks between training runs (`--lora-mode
external_peft|internal_qat`, HF-side fake-quant simulation, not a real GGUF).
To validate the exported GGUF itself rather than the simulation, run
`llama-perplexity` (build target `llama-perplexity` in the Prism
`llama.cpp` fork) directly against the produced `.gguf` file on a text
corpus — this is the only way in this pipeline to confirm the Q2_0 repack
in step 3 didn't silently corrupt anything the HF-side fake-quant eval
can't see (dtype casts, tensor naming, tied-embedding handling, etc.).
