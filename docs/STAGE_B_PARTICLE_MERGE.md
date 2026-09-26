# Stage B particle merge

Test-time Interference Search on the 1-NFE Stage B map. One page. Training design is unchanged: [VFM_FD_REDESIGN.md](VFM_FD_REDESIGN.md). Why this shape, and what not to copy, is [INTERFERENCE_VIDEO.md](INTERFERENCE_VIDEO.md).

This is an experiment protocol plus a numeric stub. It does not load weights and it does not run `RealH3Map`.

## Where it plugs in

Stage B already draws \(z \sim q_\phi(z \mid y)\) with \(y = A(x) = x[:,:,0]\), takes **one** step of the Ref2VA LoRA (`RealH3Map`, video shift 12, audio shift 3), and decodes with frozen VAEs. FD towers in `fd_loss/towers.py` (`video_tube_vjepa`, `video_frame_dinov2`, `video_frame_siglip`, `audio_clap`, `audio_beats`, `sync_imagebind`) score the population during training. Here those same frozen towers are the merge metric and the FD half of the judge. \(L_\mathrm{obs}\) stays `vfm_fd.losses.observation_loss`. FD does not replace it.

A particle is one 1-NFE decode: noise \(z\), latents, tower features, first-frame residual, FD proxy. The frontier has width \(W\). Every survivor takes the next map step on the **same** schedule. Extra budget increases \(W\), not a private \(K\) per particle.

## Algorithm

One level, matching the paper’s round (expand, execute, merge, drop seen keys, judge, keep \(W\)):

1. **Propose.** From each live particle, draw children \(z \sim q_\phi(z \mid y)\). The first level draws \(W\) noises from the adapter for the session \(y\). Text is conditioning on every draw.
2. **Execute.** One shared NFE of \(f_\theta\) for every child. Frozen VAE decode. Frozen towers. No transcript is written.
3. **Merge.** L2-normalize each tower vector, concatenate, greedy-cluster by cosine similarity at threshold \(t\) (`vfm_fd.particles.cosine_cluster`). The cluster is the soft merge key. Pixel match is not used.
4. **Drop revisits.** Quantize the representative (`quantize_key`). If that key is already in the expanded-key memory, drop the cluster.
5. **Judge.** On the representative, score \(s = -(w_\mathrm{obs}\,\|A(\hat x)-y\| + w_\mathrm{FD}\,d)\). \(d\) is a FD proxy of that particle’s features against the offline real stats or the Stage B EMA bank (distance to \(\mu_r\) is enough for the first experiment; full \(\widehat{\mathrm{FD}}\) is the training objective, not this cutoff). Dead if \(s < \tau\). Add dead keys to the memory so the mode is not retried.
6. **Keep \(W\).** Sort living clusters by \(s\). Keep one representative each, at most \(W\). Record their keys. Cluster size is a vote count, not an extra slot. Duplicates earn nothing.
7. **Advance.** Survivors are the next frontier. They take the next level together or, at the 1-NFE experiment, they are the returned set.

Stop at a fixed level count (the 1-NFE port uses one level of map calls, then optional further levels only if you are measuring a short \(K\)). Do not let one particle run ahead.

\(\tau\) is a score cutoff, not the paper’s Countdown probability. A high cutoff deletes rare good sessions. Start low enough that a holdout first-frame measurement still passes.

## Pseudocode

```text
seen ← ∅
frontier ← propose_initial(qφ, y, W)          # text is conditioning
for level in 1..L:
    children ← []
    for particle in frontier:
        children += propose(qφ, particle, y)   # z ~ qφ(z|y), or a session edit
    states ← map_one_nfe(fθ, children)         # RealH3Map, shifts 12 and 3
    feats  ← frozen_towers(decode(states))     # VAEs and towers frozen
    obs    ← observation_loss(states, y)       # ||A(fθ(z)) - y||
    proxy  ← fd_proxy(feats, real_stats)       # not a replacement for L_obs
    result ← frontier_step(feats, obs, proxy,
                           width=W, threshold=t, tau=τ, seen_keys=seen)
    seen ← result.seen_keys
    frontier ← states[result.survivor_index]
return frontier
```

`frontier_step` is `vfm_fd.particles.frontier_step`. `map_one_nfe` is the missing glue. It has to call `scripts/vfm/h3_vfm.py:RealH3Map` on a local checkpoint. The stub does not.

## Ablation versus best-of-N

Match the number of map calls. If the frontier uses \(W\) particles for \(L\) levels, best-of-\(N\) draws \(N = W \cdot L\) independent \(z \sim q_\phi(z \mid y)\), runs the same 1-NFE map, and keeps `best_of_n` (argmax of the same judge, no merge, no shared key memory).

Report, at that matched \(N\):

- First-frame residual of the returned particle.
- Per-tower feature distance (the \(\mathrm{FDr}\) vector, not one FID).
- How many clusters the merge collapsed (\(n - |\mathrm{clusters}|\)).
- Sequential levels \(L\) against \(N\) for best-of-\(N\).

A third arm, width without merge, keeps \(W\) particles and skips `cosine_cluster`. The paper’s result to test is that merge plus advancing together beats both the line and the unmerged width. If collapse is ~0, the towers are not seeing shared failures and the threshold \(t\) is wrong. If best-of-\(N\) wins on the first-frame residual, do not ship the merge.

## Training note (not this experiment)

Later, a Stage B update can score the adapter and the LoRA only on distinct clusters that stayed alive after the merge. Dead clusters and duplicates contribute zero. That is an objective note. It is not an implemented RL loop, and it does not change `vfm_nfe_loss`.

## Non-goals

- No quantum algorithm and no speedup claim.
- No tree-of-thought captions, failure ledgers, or decode-time rewind.
- No cross-attention between particles as a substitute for the merge.
- No best-first archive that never takes the next NFE.
- No unfreezing of the 33B, the VAEs, or the towers.
- No claim that `--run` generates a video. `train/stage_b_particle_merge.py --check` only checks the numeric step.

```bash
python train/stage_b_particle_merge.py --check
```
