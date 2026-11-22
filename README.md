# modular-diffusion

A flow-matching image model assembled from independently trainable units.

**Status: early.** The package is being built from a set of small prototype experiments. Interfaces will change until the
first tagged release.

## The idea

A denoiser can be cut along a few axes without coupling the pieces. Each cut is a place where a part can be taken out,
trained alone and put back.

| cut | why the pieces do not couple |
|---|---|
| by noise level | the training loss is a sum over noise levels, so a unit that is active in one stretch of the noise range shares no gradient with a unit active in another |
| by spatial-frequency band of the output | squared error is a sum over frequency bands, so units whose outputs are filtered to disjoint bands are independent at the output |
| by depth | a block whose output layers start at zero changes nothing when it is added, and can be removed exactly |
| by content | a unit gated by a declared label is silent on every other label |

## The units

| unit | what it does | how it is made |
|---|---|---|
| anchor | the best linear answer per spatial frequency, from the measured power spectrum of the data | closed form; nothing trained |
| stem, stage body | transformer blocks; a stage body is active in one window of the noise level | gradient training |
| head | maps the body's state to the answer | starts at zero |
| band member | one block and a head whose output is filtered to one frequency band | gradient training, alone |
| content expert | a unit that serves one declared label | gradient training, alone |
| growth block | a block added to a trained body, active in one window of the noise level | starts as an exact no-op |

A model is a build file (a list of units with their addresses and parents) plus the unit files. Units are linked by name.

## Conventions

```
z_t    = (1 - t) * noise + t * image        t = 0 is pure noise, t = 1 is the image
target = image - noise                      the velocity the model predicts
anchor gain at a frequency of image power S:   g = (t S - (1 - t)) / (t^2 S + (1 - t)^2)
```

## The modules

| module | what it holds |
|---|---|
| `config` | key=value arguments, read the same way from a command line and from Python keywords |
| `blocks`, `rotary` | the transformer block, zero-born layers, the time embedding; axial rotary codes for one band of wavelengths |
| `spectral` | token layout, band masks and the band filter, power spectra, the closed-form anchor |
| `windows` | windows of the noise level, hard and crossfaded |
| `units` | the unit types (anchor, stem, stage body, growth block, head, band member, content expert) and per-name birth |
| `model` | the model assembled from a build: its parts, the velocity, the training loss |
| `build` | the build file, the unit card, content hashes, unit files (safetensors + card JSON) |
| `linker` | link a build folder (every file checked against its hash), snap units in and out, the stale list; convert a model or a flat weight file into a build folder |
| `cache` | the feature cache: a frozen model's states over a training stream, so heads and members train without the stem |
| `beds`, `data` | Gaussian fields and CIFAR-style pictures; CIFAR-10 from parquet to npz |
| `train` | the recipe, the loop and the entry point (`python -m geolip.modular_diffusion.train key=value ...`) |
| `read`, `sample` | the read at fixed noise levels (by band, by part, by class); Euler sampling and a sample sheet |
| `report` | markdown tables from result files |
| `jobs` | a roster runner: one job at a time, a stop file, DONE / FAILED markers, a start guard, a wall cap |
| `hub` | upload a folder to the Hugging Face Hub with retries (token from `HF_TOKEN` only) |
| `environment` | the installed versions against `requirements.txt` |
| `compat` | the flat argument set and the single-file weight layout of the original research script |

## Notebook

| notebook | what it does | open |
|---|---|---|
| `colab/base.ipynb` | install at a tag, a 50-step smoke run, one 6,000-step base run on CIFAR-10 | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.1.0/colab/base.ipynb) |

## Install

In a hosted notebook (keeps the notebook's own torch):

```
pip install "modular-diffusion @ git+https://github.com/AbstractEyes/modular-diffusion"
```

For development and tests:

```
pip install -r requirements.txt
pip install --no-deps -e .
pytest
```

## Results

Results, weights and report pages of the prototype experiments:
https://huggingface.co/AbstractPhil/geolip-multirotary-diffusion-collective

## License

MIT.
