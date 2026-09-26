# `vfm_fd/`

Contract for the MiniMax-H3 noise adapter. Named `vfm_fd` so it does not
shadow `scripts/vfm/`. The module that trains is `H3NoiseAdapter` in
`scripts/vfm/h3_vfm.py`. This package does not redefine it.

Stage A observation is `y = A(x) = x[:, :, 0]`. `observation_loss` is
`||A(fθ(z)) - y||`. `vfm_nfe_loss` weights are mf 0.5, cos 4, std 8, obs 4,
grad 2. Endpoint `vfm_loss` is not that objective. Text passed into
`sample` is conditioning.

`sample(text, text_mask, pos, task, modality)` returns packed video and audio
rows plus diagonal-Gaussian stats (`v_mu`, `v_log`, `a_mu`, `a_log`, `kl_v`,
`kl_a`). Split rows with `split_rows` in that file. Video channels are 24,
audio channels are 32, text width is 5120, shifts are 12 and 3.

Also here:

- `unconditional_mask` / `apply_unconditional_mix` — Stage A mix, α ≈ 0.5, one mask for both streams.
- `ParameterEMA` — EMA of θ used by \(L_\mathrm{obs}\). Not the FD feature EMA.
- `observation_loss` / `vfm_nfe_loss` / `total_loss` — obs is the first-frame norm. FD is a separate term.
- `schedule` — \(K: 8 \to 4 \to 2 \to 1\). FD weight is 0 until the transition into Stage B.
- `components` — on-disk class names and attention-LoRA key patterns.

Design: `docs/VFM_FD_REDESIGN.md`.
