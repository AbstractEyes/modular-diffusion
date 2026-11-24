"""Capture: an existing diffusion U-Net (the teacher, Stable Diffusion 1.5) taken apart into band engines.

The teacher stays frozen. A band engine is a smaller U-Net made from the teacher by removing blocks in mirrored pairs,
every one of its numbers copied from the teacher, then trained alone to give the teacher's answers on one stretch of the
noise range (a band). An assembly blends the parts of the three bands by fixed weights of the step, so each part is
asked only where its band is.

    schedule    the teacher's noise schedule and the three bands, served by the library's address gate (``frame``)
    teacher     the frozen teacher (a local diffusers folder; never downloaded) and a tiny random one for tests
    shapes      engine shapes: how many layers each stage keeps, the middle block, three or four stages
    engine      an engine built from the teacher by paired block removal, its numbers inherited exactly
    loss        the capture loss: squared error to the teacher's answer, and optionally to the noise
    train       one band engine trained alone, with an averaged copy and exact resume
    assemble    the assembly and a deterministic DDIM sampler with guidance
    reads       fixed probes and the reads: a band's read, the born read, the exchange test, the off-band control
    pictures    an assembly's pictures (sampled group by group, decoded) and their scores: a kernel distance, a
                text-to-picture judge, a contact sheet

Conventions: steps t = 0 .. 999, larger is noisier; z_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps; the teacher predicts eps.

TWO DIRECTIONS OF t LIVE IN THIS PACKAGE. Here t counts the teacher's way: t = 0 is the picture's end, t = 999 the
noise's. The rest of the package (the staged models and their experiments) counts the other way, on the unit interval:
t = 0 is pure noise, t = 1 the picture, z_t = (1 - t) noise + t picture. No function takes both. A band's coordinate
here is s = t / 1000 (larger is noisier), so the band named ``clean`` sits at small s; a stage window of the staged
models that sits near THEIR t = 1 covers the same end of the trip.

The optional packages (diffusers, transformers) are imported inside these modules only: ``pip install
modular-diffusion[capture]``.
"""

BANDS = ("clean", "middle", "noisy")
