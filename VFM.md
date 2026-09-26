# VFM × FD-loss

Few-step Ref2VA for this fork. The full design is [docs/VFM_FD_REDESIGN.md](docs/VFM_FD_REDESIGN.md). Upstream MiniMax H3 checkpoints and docs are unchanged.

Three roles, not one model:

- **Adapter** \(q_\phi(z \mid y)\) — `scripts/vfm/h3_vfm.py` `H3NoiseAdapter`. Picks initial video/audio noise that fits the reference, the identity, and the spoken line.
- **LoRA** on the Ref2VA Omni-Transformer — the flow map \(f_\theta\). `MiniMaxH3DiTModel` under `Ref2VA/transformer`, or `MiniMaxH3Transformer3DModel` under `transformer_ref/`. Thin attention LoRA only. Not the 33B, not the VAEs.
- **FD-loss** — population Fréchet penalty in frozen video, audio, and sync towers. This is what Stage B uses to post-train the LoRA at **1 NFE**. It does not learn the conditioning.

Stage A locks constraints at \(K \in \{4, 2\}\). Stage B freezes the adapter and post-trains the LoRA on 1-NFE samples from \(q_\phi\), which is how a single step is meant to reach the real AV distribution. Stage C is an optional coarse FD on 17-frame canvases (`clip_length` / `vae_clip_length` 17).

Scaffolds (no weight download, no GPU training):

```bash
python train/stage_a_vfm.py --check
python train/stage_b_fd_turbo.py --check
python train/stage_c_canvas_fd.py --check
python -m fd_loss.offline_real_stats --synthetic --out /tmp/h3-fd-stats
```

The existing toy falsifier is still `python scripts/vfm/h3_vfm.py` (it needs PyTorch).
