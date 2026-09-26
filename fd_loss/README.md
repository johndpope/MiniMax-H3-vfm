# `fd_loss/`

Population Fréchet penalty on frozen representation towers. Used by Stage B
(1-NFE LoRA post-training) and, optionally, Stage C canvases.

- `stats.py` — `EMAFeatureStats` (β ≈ 0.999) and `QueueFeatureStats` (default 8192, range 8k–32k). `surrogate` then `commit`. Only the current batch is the live branch.
- `frechet.py` — closed-form FD, plus a frozen per-tower scale (FD against N(0, I)).
- `towers.py` — video tubes (V-JEPA / InternVideo), frames (DINOv2, SigLIP), audio (CLAP, BEATs), sync (ImageBind or a small AV head).
- `multi_tower.py` — weighted sum of normalized tower FDs. The return value keeps the per-tower vector.
- `offline_real_stats.py` — write \((\mu_r, \Sigma_r)\) once. `--synthetic` needs no weights. The real encode path is a TODO and must not load the 33B.

```bash
python -m fd_loss
python -m fd_loss.offline_real_stats --synthetic --out /tmp/h3-fd-stats.npz
```

Design: `docs/VFM_FD_REDESIGN.md`.
