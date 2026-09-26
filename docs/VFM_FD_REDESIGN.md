# VFM × FD-loss redesign

Few-step Ref2VA on MiniMax-H3. The noise adapter picks the initial noise for a reference session. A thin LoRA is the flow map. FD-loss is the post-training objective that is meant to make a **1-NFE** sample match the real audio-video distribution.

This document is the design. Runnable training is not implemented here. The numeric pieces that do not need weights live under `vfm_fd/` and `fd_loss/`. Stage entry points live under `train/`. The scaffold package is `vfm_fd` so a repo-root `import vfm` cannot shadow `scripts/vfm/`.

The released H3 tree stays where it is: `FL2VA/`, `Ref2VA/`, `vae/`, `audio_vae/`, `transformer/`, `transformer_ref/`. Nothing in this redesign replaces those checkpoints.

## Central bet

Ref2VA stays a session: block the shot, lock identity, tune the spoken line on the audio canvas, then spend fine steps. VFM chooses the noise for that session. FD-loss makes the cheap fine path good enough that the expensive many-step path is rare.

Stage B is the speed lever. After Stage A has a conditional proposal \(q_\phi(z \mid y)\) whose few-step map already matches the first frame, Stage B draws \(z\) from that adapter and post-trains only the LoRA so that **one** integration of \(f_\theta\) lands in the real AV feature cloud. Extra solver steps are then a fallback when the first-frame measurement slips, not the default way to buy quality.

This repo has not measured 1-NFE quality of the released 33B weights. The stages below are the training plan. They are not a diagnosis of the released sampler.

## Roles

Three pieces, kept separate:

| Piece | Symbol | Owns | Does not own |
|---|---|---|---|
| VFM noise adapter | \(q_\phi(z \mid y)\) | Initial noise for the first-frame measurement | The transport from noise to latents. Population look of the decoded clip |
| H3 flow / LoRA | \(f_\theta\) | Map \(z \to\) AV latents in \(K\) NFEs, annealed toward 1 | The measurement \(y\). Text does not stand in for \(y\) |
| FD-loss | \(\widehat{\mathrm{FD}}_{\varphi_i}\) | Population match of low-NFE outputs to the real AV distribution in frozen representation towers | The first-frame measurement, or any per-sample constraint |

\(y = A(x) = x_{:,:,0}\), the I2VA first frame. Time is axis 2, so \(A\) is `x[:, :, 0]`. Text and Qwen3-VL states (`text_dim` 5120) condition the adapter and the map. They are not \(y\). \(z\) is packed video+audio noise. \(x = f_\theta(z)\) is the packed video latent (24 channels) and audio latent (32 channels).

The in-repo adapter is `H3NoiseAdapter` in `scripts/vfm/h3_vfm.py`. \(q_\phi\) is a diagonal Gaussian, the same family the script samples at \(t = 1\):

\[
z = \mu_\phi(y) + \exp(\log \sigma_\phi(y)) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I).
\]

That class is the one Stage A trains. `vfm_fd/adapter.py` is the contract around it (sample signature, unconditional mix, KL). It is not a second network. The VAE posterior in `Ref2VA/video_vae/vae_module.py` (`DiagonalGaussianDistribution`) is a different Gaussian and is not \(q_\phi\).

## What already exists

`scripts/vfm/h3_vfm.py`:

- `H3NoiseAdapter.sample(text, text_mask, pos, task, modality, temperature=1.0)` returns packed rows plus `v_mu`, `v_log`, `a_mu`, `a_log`, `kl_v`, `kl_a`. Video rows are the first \(S_v\) of the video head. Audio rows are \(S_v : S_v+S_a\) of the audio head. `split_rows` is the splitter. Modality 0 is video, 1 is audio. The `text` argument is conditioning.
- `shift_sigma` implements the released shifts: video 12, audio 3.
- `kl_to_standard_normal` is the KL term.
- `vfm_loss` is endpoint latent MSE, \(x = z + v\) at \(t = 1\). For a velocity head, \(\|v - (x_0 - z)\|^2 = \|\hat x - x_0\|^2\), so \(v = -z\) is a black frame and plain MSE prefers that basin. Stage A does not use this loss and does not generalize it.
- Stage A uses `vfm_nfe_loss` with weights mf 0.5, cos 4, std 8, obs 4, grad 2 (`vfm_fd.losses.NFE_WEIGHTS`). The obs term is `observation_loss`, \(\|A(f_\theta(z)) - y\|\).
- `RealH3Map` is the map the trainer calls. Stage A does not add a task to reimplement `velocity`.
- Task ids: `t2va=0`, `i2va=1`, `fl2va=2`, `ref2va=3`, `talking_head=4`.

v1 training targets the Ref2VA partition. FL2VA stays on disk and is not a training target.

## H3 components the stages attach to

Two on-disk layouts, one model. LoRA patterns follow the **weight keys actually indexed in this repo**.

| Role | Original checkpoint | Diffusers layout |
|---|---|---|
| Ref2VA flow | `Ref2VA/transformer`, class `MiniMaxH3DiTModel` (`blocks.*`) | `transformer_ref/`, class `MiniMaxH3Transformer3DModel` (`transformer_blocks.*`) |
| FL2VA flow (not a v1 target) | `FL2VA/transformer`, `MiniMaxH3DiTModel` | `transformer/`, `MiniMaxH3Transformer3DModel` |
| Video VAE | `Ref2VA/video_vae`, `MiniMaxH3VideoVAE` / `AutoencoderKLLegacy`, `vae_clip_length` 17, 24 latent channels, f16t4d24 | `vae/`, `AutoencoderKLMiniMaxH3`, `clip_length` 17 |
| Audio VAE | `Ref2VA/audio_vae`, `MiniMaxH3AudioVAE`, 32 channels, `sample_rate` 32000 | `audio_vae/`, `AutoencoderKLMiniMaxH3Audio`, `sampling_rate` 32000 |
| Shifts | `Ref2VA/model_index.json` `_minimax_h3.sigma_shift_scales` video 12, audio 3 | `scheduler/scheduler_config.json` shift 12, `audio_scheduler/scheduler_config.json` shift 3, class `MiniMaxH3Scheduler` |
| Text | `Ref2VA/text_encoder`, `MiniMaxH3Qwen3VLHFEncoder` | `text_encoder/`, `Qwen3VLForConditionalGeneration` |
| Pipeline | `Ref2VA` `MiniMaxH3Pipeline` | root `MiniMaxH3ModularPipeline` |

Thin LoRA v1, attention only:

- Original: `blocks.*.attn.qkv_proj`, `blocks.*.attn.out_proj`. Optional later: `blocks.*.mlp.fc1`, `blocks.*.mlp.fc2`.
- Diffusers: `transformer_blocks.*.attn.to_q`, `to_k`, `to_v`, `to_out.0`. Optional later: `transformer_blocks.*.ff.net.0.proj`, `transformer_blocks.*.ff.net.2`.

Leave frozen in v1: every `adaln_proj` (the README attributes ~13B of the 33B to AdaLN), `proj_in` / `audio_proj_in` / `proj_out` / `audio_proj_out` (diffusers) and `audio_patch_proj` (original), token refiners, both VAEs, the text encoder, and every representation tower. Rank is a hyperparameter (`--lora-rank`, scaffold default 16). It is not a result from a paper.

Few-step \(K\) uses the released shifts: video \(\sigma\)-shift 12, audio \(\sigma\)-shift 3, through `RealH3Map`. Do not invent a second noise schedule. The production step size should follow `MiniMaxH3Scheduler` once that class is imported. Do not swap that call for endpoint `vfm_loss`.

## Stage A — constraint lock

Train `H3NoiseAdapter` and a thin LoRA on the Ref2VA Omni-Transformer.

- Objective: `vfm_nfe_loss` with weights mf 0.5, cos 4, std 8, obs 4, grad 2. FD weight is 0.
- Observation: \(y = A(x) = x_{:,:,0}\). `vfm_fd.losses.observation_loss` is \(\|A(f_\theta(z)) - y\|\). Text and Qwen states are conditioning.
- Entry \(K \in \{4, 2\}\), not 1. Optional warmup \(K = 8\) exists on the schedule and is off unless requested.
- Mix unconditional noise with \(\alpha \approx 0.5\): each example is either drawn from the adapter or from \(\mathcal{N}(0, I)\), with probability \(\alpha\) on the standard normal. One Bernoulli mask covers video and audio together. This is `vfm_fd.adapter.unconditional_mask` / `apply_unconditional_mix`. It is not a variance blend.
- Freeze both VAEs, the text encoder, and every representation tower.
- The map inside the observation is \(\theta_\mathrm{EMA}\) (decay 0.999, `vfm_fd.ema.ParameterEMA`), updated after the optimizer step.

Do not train endpoint `vfm_loss` and do not widen it to \(K\) steps. That MSE is the black-basin objective above.

## Stage B — FD turbo

This is the stage that is meant to unlock 1-NFE quality.

1. Load the Stage A adapter and LoRA. Freeze the adapter (`--adapter-lr` default 0; a tiny LR is optional and is the only Stage B path that still trains \(\phi\)).
2. Draw \(z \sim q_\phi(z \mid y)\) and integrate **\(K = 1\)** with the online LoRA, shifts 12 / 3.
3. Decode with the frozen video and audio VAEs. Run frozen towers on the decode. Do not detach that forward: tower and VAE **weights** stay frozen, but the FD gradient has to reach the LoRA through the decode. Detaching the VAE kills the objective.
4. Post-train the LoRA with EMA FD-loss, \(\beta \approx 0.999\), on those 1-NFE features. Backprop only through the current batch. Population state is a constant.
5. Keep logging \(L_\mathrm{obs}\). If constraint metrics move the wrong way, stop or restore a non-zero adapter LR. FD is not allowed to become the only objective.
6. Real moments \((\mu_r, \Sigma_r)\) are computed once, offline, on real Ref2VA-domain clips. See `fd_loss/offline_real_stats.py`. The 33B transformer is not part of that job.

Why a single step can be enough if this works: the adapter has already moved \(z\) to a noise point from which the Stage A map matches the first frame in a few NFEs. The remaining error at 1 NFE is distributional, which is what a population Fréchet penalty on the decoded features can see. FD does not see \(y\). The adapter remains the module that maps the first frame to \(z\). Text stays conditioning.

Population, pick one per run and record it:

- **EMA** (brief default): \(\beta = 0.999\). `fd_loss.stats.EMAFeatureStats`. The mean is the usual blend. The covariance is the mixture
  \(\Sigma \leftarrow \beta\Sigma + (1-\beta)\Sigma_b + \beta(1-\beta)(\mu - \mu_b)(\mu - \mu_b)^\top\),
  where \(\mu\) is the EMA mean before the batch. The previous EMA state is held fixed while the surrogate is built. The batch share of the mean is \(1 - \beta = 0.001\), so the FD gradient is small. That is a property of this estimator, not a reason to unfreeze the 33B.
- **Queue** (alternate): 8k–32k feature slots, exact mixture mean and covariance of the detached queue plus the current batch. `fd_loss.stats.QueueFeatureStats`. Default capacity on the scaffold is 8192. The batch's share of the moment is \(n_\mathrm{batch} / (n_\mathrm{queue} + n_\mathrm{batch})\).

Commit the batch into the population only after the surrogate moments are formed, and commit detached features. Otherwise the batch is counted twice or the queue itself becomes a graph.

### Towers

Multi-tower mix, each FD normalized before the weighted sum (FD-SIM style). Report the vector. Do not collapse it to one FID.

| Name | Family | What it sees | Scaffold weight |
|---|---|---|---|
| `video_tube_vjepa` | V-JEPA tubes. Alternate: InternVideo | video tubes | 0.25 |
| `video_frame_dinov2` | DINOv2 frames | frames | 0.15 |
| `video_frame_siglip` | SigLIP frames | frames | 0.15 |
| `audio_clap` | CLAP | audio clips | 0.15 |
| `audio_beats` | BEATs | audio clips | 0.10 |
| `sync_imagebind` | ImageBind. Alternate: a small AV-sync head | paired AV | 0.20 |

Weights sum to 1. They are a starting mix in `fd_loss/towers.py`, not a published recipe. A tower with no offline stats does not get a silent zero; the loss raises until you disable it.

Normalization: divide each tower's FD by a **frozen** positive scale. The scaffold scale is \(\max(\mathrm{FD}(\mathrm{real}, \mathcal{N}(0, I)), \varepsilon)\), computed once from \((\mu_r, \Sigma_r)\) and stored. Swap in the paper's published per-tower normalizer when you pin that formula. Do not re-estimate the scale from the current batch.

\(\mathrm{FDr}^k\) in this design is that per-tower representation Fréchet distance. \(k\) indexes the tower. Eval uses a fresh generated population at fixed 1 NFE, not the training EMA or queue.

## Stage C — canvas FD

Optional. Use it only if 17-frame chunks are exposed as editable boards.

`clip_length` / `vae_clip_length` is 17 on both video VAE configs (causal clip \(1 + 4 \times 4\), matching f16t4 temporal compression). A canvas is one decoded 17-frame chunk, spatially pooled so the FD sees layout, parts, and sync before fine detail.

Compare student canvases to teacher canvases from **one** source per stats file: real Ref2VA-domain clips, or frozen high-NFE decodes of the same \(y\). Do not mix those sources in one \((\mu_r, \Sigma_r)\). The canvas term is a light extra weight on the same frozen-tower FD interface. It does not replace Stage B and it does not replace \(L_\mathrm{obs}\).

## Step anneal

One schedule. FD weight stays at 0 through the constraint lock, then rises as \(K\) falls.

| Phase | \(K\) | FD weight | Trains |
|---|---|---|---|
| Optional warmup | 8 | 0 | adapter + thin LoRA |
| Stage A early | 4 | 0 | adapter + thin LoRA |
| Stage A late | 2 | 0 | adapter + thin LoRA |
| Transition | 2 | rising (scaffold start 0.1 of the Stage B weight) | LoRA; adapter LR decaying |
| Stage B | 1 | full | LoRA; adapter frozen or tiny LR |

Stop when the 1-NFE \(\mathrm{FDr}\) vector plateaus **and** the first-frame measurement \(\|A(f_\theta(z)) - y\|\) still holds relative to the Stage A checkpoint. A better FD with a worse first frame is a failed run.

`vfm_fd.schedule.fd_weight_for_k` encodes the table. Stage A defaults to \(K = 4\).

## Loss sketch

Stage A is `vfm_nfe_loss`, not endpoint MSE:

\[
L_\mathrm{NFE} = 0.5\, L_\mathrm{mf} + 4\, L_\mathrm{cos} + 8\, L_\mathrm{std} + 4\, L_\mathrm{obs} + 2\, L_\mathrm{grad}
\]

\[
L = L_\mathrm{NFE} + \lambda_\mathrm{FD} \sum_i w_i \widehat{\mathrm{FD}}_{\varphi_i}
\]

- \(L_\mathrm{obs} = \|A(f_\theta(z)) - y\|\) with \(y = A(x) = x_{:,:,0}\). Not an FD. Not a text embedding.
- \(\lambda_\mathrm{FD} = 0\) in Stage A and rises on the anneal. Stage B uses the full weight at \(K = 1\).
- \(\widehat{\mathrm{FD}}_{\varphi_i}\): normalized tower FD. Weight \(w_i\) from the table.
- FD is computed on features of the **current** decoded latents. Towers and VAEs stay frozen.
- `vfm_loss` in `scripts/vfm/h3_vfm.py` stays the endpoint MSE. Stage A does not call it and does not generalize it.

`vfm_fd.losses.total_loss` adds the FD term on top of `vfm_nfe_loss`. Stage A passes FD weight 0.

Gradient contract for the FD term (`fd_loss.stats` surrogate, then `commit`):

- Detached: real \((\mu_r, \Sigma_r)\), the population accumulated before this batch, tower weights, VAE weights.
- Live: the current batch of tower features, which depend on the decode of the current latents, which depend on the LoRA (and on \(\phi\) only if the adapter LR is non-zero).

## FD-loss mechanics

Fréchet distance between two Gaussians:

\[
\|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2 \left(\Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2}\right)^{1/2}\right).
\]

`fd_loss.frechet.frechet_distance` is this expression, with a ridge and an eigenvalue clip on the symmetric square roots. It is the training surrogate and the eval \(\mathrm{FDr}^k\). No GPU and no tower weights are required to exercise it.

Offline real stats (`python -m fd_loss.offline_real_stats`):

- Input: a manifest of real Ref2VA-domain clips.
- Encode with frozen towers only. Do not load `MiniMaxH3DiTModel` / `MiniMaxH3Transformer3DModel`.
- Write one \(\mu_r\), \(\Sigma_r\), count, and dim per tower, plus the frozen norm scale.
- `--synthetic` writes moments from a local RNG so the plumbing can be checked. It does not download anything. The real tower forward is a TODO at `encode_manifest`.

## Eval checklist

Run at **fixed 1 NFE** unless a row says otherwise. Log the vector. Do not average it into one score for the stop rule.

Constraints (must hold, not just the FD):

- Identity: subject / face similarity of the sample against the Ref2VA reference frames for that \(y\).
- Line: the spoken line matches the conditioned dialogue (ASR or an audio embedding against the reference line).
- Sync: lip sync / AV alignment on the same sample. The sync tower used in training can be one signal; a held-out sync metric should be another, so the training tower is not grading itself alone.
- Dual-head health: video and audio latent error, both KL terms, both \(\|\mu\|\) from the adapter. Audio must stay in the same ballpark as video on the sessions you care about.
- Stage A reference: constraint numbers at 1 NFE are compared to the Stage A checkpoint at its exit \(K\) and at 1 NFE. FD is not allowed to buy its score by dropping these.

Population (1 NFE, fresh samples, offline real stats):

- \(\mathrm{FDr}\) for `video_tube_vjepa`, `video_frame_dinov2`, `video_frame_siglip`.
- \(\mathrm{FDr}\) for `audio_clap`, `audio_beats`.
- \(\mathrm{FDr}\) for `sync_imagebind`.
- The same vector at the exit \(K\) of Stage A, once, as a baseline. The claim to test is the 1-NFE vector, not a longer sampler.
- Optional Stage C: canvas \(\mathrm{FDr}\) on pooled 17-frame boards, reported next to the full-detail vector, not instead of it.

Stop when the 1-NFE vector plateaus and the constraint rows still hold. Keep the many-step sampler. It is the fallback the central bet is trying to make rare.

## Anti-patterns

- Do not replace \(L_\mathrm{obs}\) with FD. \(L_\mathrm{obs}\) is the first-frame measurement. FD does not know \(y\).
- Do not FD-train the full 33B. Train the adapter and a thin LoRA only.
- Do not chase a single FID. Report the multi-tower \(\mathrm{FDr}\) vector (video, audio, sync).
- Do not expect FD to invent the measurement. The adapter maps \(y = x_{:,:,0}\) to \(z\). Text is conditioning.
- Do not unfreeze VAEs or towers in v1. Do not detach the decode either, or the LoRA gets no FD gradient.
- Do not point \(q_\phi\) at the VAE's `DiagonalGaussianDistribution`.
- Do not mix real clips and many-step teacher decodes in one real-stats file.
- Do not download weights from these scaffolds. Local directories only, and only after the TODOs in `train/stage_*.py` are filled in.

## Scaffold map

| Path | What it is |
|---|---|
| `scripts/vfm/h3_vfm.py` | Adapter, shifts, endpoint `vfm_loss` (not Stage A), `RealH3Map` |
| `vfm_fd/` | Adapter contract, first-frame `observation_loss`, `vfm_nfe_loss`, \(\theta\) EMA, anneal |
| `fd_loss/` | EMA and queue stats, Fréchet, multi-tower mix, offline real-stats stub |
| `train/stage_a_vfm.py` | Stage A argparse and TODOs |
| `train/stage_b_fd_turbo.py` | Stage B argparse and TODOs |
| `train/stage_c_canvas_fd.py` | Optional canvas FD argparse and TODOs |
| `VFM.md` | Short pointer |

`python train/stage_a_vfm.py --check` (and the Stage B / C equivalents) runs the numeric self-checks and the on-disk wiring checks. `--run` stops on the TODO. It does not load a checkpoint.

## Out of scope

Full training, LoRA injection, tower implementations (DINOv2, V-JEPA, InternVideo, SigLIP, CLAP, BEATs, ImageBind), and any download of H3 or tower weights. H3-Context-IR and H3-Regenerate-2K stay hosted components, as in the upstream README.
