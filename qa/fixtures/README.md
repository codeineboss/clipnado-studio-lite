`nasa-astronaut.png`: astronaut Eileen Collins, NASA public-domain photograph,
as distributed by scikit-image v0.22.0 (`skimage/data/astronaut.png`).
Source: https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/data/astronaut.png
SHA-256: 88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5

Used only to generate synthetic QA footage for face tracking and executing the
actual models. A static portrait with a tone is NOT a speaking-person accuracy
benchmark and is not presented as one. It is not an application user video.

`mfcc-reference.npz`: our generated two-tone PCM16 signal and golden MFCC
features independently evaluated with `python_speech_features==0.6` (MIT) and
`scipy==1.15.3`, using default MFCC parameters at 16 kHz. These libraries are
build-only references, not shipped runtime dependencies. Array file is loaded
with `allow_pickle=False`.
