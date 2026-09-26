"""Default multi-tower mix. Weights are a starting point, not a published recipe.

Video tubes: V-JEPA (alternate InternVideo). Video frames: DINOv2 and SigLIP.
Audio: CLAP and BEATs. Sync: ImageBind (alternate: a small AV-sync head).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TowerSpec:
    name: str
    family: str
    modality: str  # video, audio, sync
    view: str  # tube, frame, clip, pair
    weight: float
    feature_dim: int | None = None
    norm_scale: float | None = None  # frozen FD-SIM divisor; None until stats exist

    def with_scale(self, norm_scale: float, feature_dim: int) -> "TowerSpec":
        if norm_scale <= 0:
            raise ValueError(f"{self.name} norm_scale must be positive")
        return replace(self, norm_scale=float(norm_scale), feature_dim=int(feature_dim))


DEFAULT_TOWERS: tuple[TowerSpec, ...] = (
    TowerSpec("video_tube_vjepa", "vjepa", "video", "tube", 0.25),
    TowerSpec("video_frame_dinov2", "dinov2", "video", "frame", 0.15),
    TowerSpec("video_frame_siglip", "siglip", "video", "frame", 0.15),
    TowerSpec("audio_clap", "clap", "audio", "clip", 0.15),
    TowerSpec("audio_beats", "beats", "audio", "clip", 0.10),
    TowerSpec("sync_imagebind", "imagebind", "sync", "pair", 0.20),
)

ALTERNATES = {
    "video_tube_vjepa": "internvideo",
    "sync_imagebind": "av_sync_head",
}


def enabled_towers(
    names: list[str] | None = None,
    towers: tuple[TowerSpec, ...] = DEFAULT_TOWERS,
) -> tuple[TowerSpec, ...]:
    """Subset of towers, weights renormalized to sum to 1."""
    if names is None:
        chosen = towers
    else:
        wanted = set(names)
        unknown = wanted.difference(spec.name for spec in towers)
        if unknown:
            raise KeyError(f"unknown towers {sorted(unknown)}")
        chosen = tuple(spec for spec in towers if spec.name in wanted)
        if not chosen:
            raise ValueError("no towers selected")
    total = sum(spec.weight for spec in chosen)
    return tuple(replace(spec, weight=spec.weight / total) for spec in chosen)
