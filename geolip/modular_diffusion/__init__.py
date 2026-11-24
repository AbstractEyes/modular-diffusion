"""modular-diffusion: a flow-matching image model assembled from independently trainable units.

Conventions used throughout the package:

    z_t    = (1 - t) * noise + t * image        t = 0 is pure noise, t = 1 is the image
    target = image - noise                      the velocity the model predicts
"""

__version__ = "0.10.0"
