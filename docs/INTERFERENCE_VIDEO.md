# Interference Search for low-NFE video: merge in FD-tower space

This note is why a test-time particle merge sits next to the VFM × FD redesign. It does not change Stage A, the FD loss, or the package name `vfm_fd`. The experiment itself is [STAGE_B_PARTICLE_MERGE.md](STAGE_B_PARTICLE_MERGE.md). Training design stays [VFM_FD_REDESIGN.md](VFM_FD_REDESIGN.md).

## Why this file exists

Stage B draws one noise \(z \sim q_\phi(z \mid y)\) and takes one step of the LoRA. Independent draws of that noise share the adapter’s biases, so extra samples often fail the same way: the same identity drift, the same desync. Spending the budget as \(W\) separate 1-NFE videos does not remove a failure that every particle hits.

Interference Search is a different shape for that budget. It was written for explicit states, not for diffusion. The piece that transfers is the shape: expand a frontier together, merge branches that land on the same situation, cancel dead ends with a small judge, and advance the survivors one level at a time. The name is a quantum-search metaphor. The method is classical. The paper claims no quantum speedup.

## The paper

Al-ameen, *Interference Search: Reasoning over merged states, many branches at once*, Bad Theory Labs, Lagos, September 2026.

- Paper: https://www.badtheorylabs.com/papers/interference-search/paper.pdf
- Code and data: https://github.com/Badtheorylabs/interference-search (Apache-2.0)

A domain supplies six functions: a start state, a proposer, a merge key, a judge, a goal test, and a dead-end test. One round expands every live state, the environment executes the moves, equal states merge, keys already expanded are dropped, the judge scores what remains, and the best \(W\) advance one level together. Nothing carries between rounds except the states and the set of merge keys already expanded. \(W\) comes from the budget, so extra budget widens the frontier. Depth stays the depth of the problem.

On 30 hard four-number Countdown problems, with the same judge, 200 judged positions solve 30 of 30 for the frontier and 21 of 30 for one line of thought. The line needs 1,500 judged positions, and 23.7 sequential steps on average, to solve all 30. The frontier’s depth on those solves is 3. At 100 expansions on 100 unseen six-number problems, the merged frontier solves 77%, the same frontier without merging 50%, one line 40%, and a global best-first memory 10%. The paper’s reading is that the gain is merge plus level-synchronous advance, not a ranked list of every state ever seen, and not streams that merely attend to each other.

The paper does not train the language model. That is stated as the next step. Code results on 30 MBPP first-try failures are within noise (9 solved against 8 for fresh best-of-\(N\) samples).

## What this stack already is

From [VFM_FD_REDESIGN.md](VFM_FD_REDESIGN.md):

- **Adapter** \(q_\phi(z \mid y)\), `H3NoiseAdapter` in `scripts/vfm/h3_vfm.py`. \(y = A(x) = x[:,:,0]\) (time on axis 2). Text and Qwen states condition the model. They are not \(y\).
- **LoRA** \(f_\theta\) on the Ref2VA Omni-Transformer (`MiniMaxH3DiTModel` or `MiniMaxH3Transformer3DModel`). Short \(K\)-step map. Shifts stay video 12 and audio 3. The 33B base, both VAEs, and the towers stay frozen.
- **FD-loss** in `fd_loss/`. Multi-tower population Fréchet: video tubes (`video_tube_vjepa`, alternate InternVideo), frames (`video_frame_dinov2`, `video_frame_siglip`), audio (`audio_clap`, `audio_beats`), sync (`sync_imagebind`, alternate a small AV head). FD never replaces \(L_\mathrm{obs}\). \(L_\mathrm{obs}\) is `vfm_fd.losses.observation_loss`, \(\|A(f_\theta(z)) - y\|\).
- Stage A locks that measurement at \(K \in \{4, 2\}\) with `vfm_nfe_loss` (mf 0.5, cos 4, std 8, obs 4, grad 2) and FD weight 0. Stage B is 1-NFE FD post-training of the LoRA. Stage C is optional coarse FD on 17-frame canvases.

Particle merge does not replace any of that. It is a way to spend a test-time budget of map calls.

## Six functions on this stack

| Paper function | Here |
|---|---|
| Start | One Ref2VA session: conditioning text, reference images / video / audio, first-frame \(y\), remaining step budget, and the initial noise the adapter proposes. |
| Proposer | \(q_\phi\). The Stage B experiment draws several \(z\) for the same \(y\). A wider session search can also re-roll a region, swap a reference, change \(K\), or accept or reject a noise. Those moves are still proposals. The environment, not a caption, applies them. |
| Merge key | A soft key in frozen tower space. Concatenate L2-normalized features from the FD towers and cluster by cosine similarity. Pixel equality is the wrong key: two decodes of the same failure rarely match bitwise. This is the same idea as FETCH’s embedding merge of semantically equivalent search states, which the paper cites, not a new tower. |
| Judge | A cheap score, not the 33B. First-frame residual from `observation_loss`, plus a FD proxy of that particle’s tower features against the offline real \((\mu_r, \Sigma_r)\) or the Stage B EMA bank. Mid-trajectory sync can be an extra probe. The cutoff \(\tau\) is on this score. |
| Goal | At the budgeted NFE, the first-frame measurement still holds and the multi-tower \(\mathrm{FDr}\) vector is the one you would have logged for Stage B. Not a single FID. |
| Dead | Score below \(\tau\), or a merge key already expanded. One cancellation drops every particle that landed in that cluster, including ones that were never written out as text. |

The environment is H3: one shared schedule step of `RealH3Map` (shifts 12 and 3), then a frozen VAE decode (`MiniMaxH3VideoVAE` / `AutoencoderKLMiniMaxH3`, `MiniMaxH3AudioVAE` / `AutoencoderKLMiniMaxH3Audio`) into the frozen towers. All live particles take that step together. The search does not carry a transcript.

Two ports, in the order to try them:

1. **1-NFE particles (Stage B).** Proposer is the adapter. One level is one NFE. Merge and judge the decodes. This is the sketch in [STAGE_B_PARTICLE_MERGE.md](STAGE_B_PARTICLE_MERGE.md).
2. **Session search (Ref2VA).** State is the canvas plus refs, audio, and budget. Moves are edit operations. Merge canvases that match under the towers. Same judge. Do this only after the 1-NFE merge is measured.

## What does not transfer

Countdown has exact states and an exact dead-alive oracle from dynamic programming. Video does not. There is no free equality test and no free label for “this decode can still hit \(y\)”. The merge key has to be learned or at least tuned in tower space, and the judge has to be trained from rollouts or from Stage A constraint logs. The paper’s limitation section says the same thing for open domains.

Aggressive \(\tau\) deletes rare good paths. On random six-number Countdown the judge at \(\tau = 0.9\) kept 86% of solutions; on hard instances with at most 24 solution paths, \(\tau = 0.7\) kept 43%. A tight FD cutoff will do the same to an unusual but valid Ref2VA session. Start permissive. Raise \(\tau\) only while the first-frame measurement on a holdout still holds.

Width without merge burns NFEs. The no-merge frontier in the paper is only a little above one line, because duplicate states fill \(W\). Shared failure modes here are identity drift and desync. If every particle is the same drift, merge should collapse them to one slot and free the rest of \(W\). If you skip the merge, you have paid for \(W\) copies of one mistake.

## Negative results that constrain the mechanism

The paper tried to get the same behaviour out of a small model without a state merge. Those arms failed, and they bound what we should not build.

- **Textual failure ledgers.** Telling streams which expressions were already refuted made them retry those expressions. Within 80 tokens of a note, 33.6% of attempts repeated a sibling’s refutation, against about 21% with no note. Solve rate did not move. Do not bolt tree-of-thought captions, or a list of bad prompts, onto Ref2VA.
- **Decode-time rewind.** Rewinding a stream that wrote a known-bad expression regenerated the same line, up to 16 times, because the cause was still in the context. Do not “fix” a bad 1-NFE by resampling tokens of a caption. Change the noise or the canvas state, then merge.
- **Cross-stream attention.** Policies that could see sibling streams tied policies that could not (0.508 and 0.511). Both lost to a player with an explicit shared memory (0.82). Parallel tower features that only attend to each other are not a merge. Coordination has to be the merge and the judge.
- **Best-first over every state ever scored.** That arm solved 10% where the level-synchronous frontier solved 77% at the same expansion count. A global buffer of pretty frames that never commits to the next NFE is the same mistake. Survivors take one map step together.

Training inside the frontier is the paper’s suggested next step, and it is not done there. Duplicates merge and earn nothing. Dead clusters score zero. A model trained that way is rewarded for distinct live branches. On this stack that would be a later Stage B objective for the adapter and the LoRA, not an RL run this repo claims to have.

## What to read next

Implement the test-time loop from [STAGE_B_PARTICLE_MERGE.md](STAGE_B_PARTICLE_MERGE.md). The numeric merge and the judge cutoff live in `vfm_fd/particles.py`. `train/stage_b_particle_merge.py --check` exercises them. `--run` stops before any DiT call.
