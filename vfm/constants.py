"""Shapes and shifts taken from the on-disk H3 configs and the toy adapter.

Kept in sync with ``scripts/vfm/h3_vfm.py`` (``H3_TEXT_DIM``, channel counts,
shifts, ``TASK``). ``train`` checks that the script still defines the same
literals.
"""

H3_TEXT_DIM = 5120
H3_VIDEO_CH = 24
H3_AUDIO_CH = 32
H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0

# Causal video-VAE clip. vae/config.json clip_length and
# Ref2VA/video_vae/config.json vae_clip_length.
VAE_CLIP_LENGTH = 17

# scripts/vfm/h3_vfm.py TASK. Ref2VA is the v1 training target.
TASK = {
    "t2va": 0,
    "i2va": 1,
    "fl2va": 2,
    "ref2va": 3,
    "talking_head": 4,
}
