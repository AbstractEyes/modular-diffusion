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
| body | several blocks after the stem, on every noise level or (windowed) mixed by the stage weights | gradient training |

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
| `spectral` | token layout, band masks and the band filter, power spectra, the closed-form anchor, the per-class anchor (a mean field and a spectrum per class) and its statistics |
| `windows` | windows of the noise level, hard and crossfaded |
| `units` | the unit types (anchor, optionally per class; stem, optionally reading the class only in a window of the noise level; stage body, growth block, body, depth block (a block appended before the head, the exact identity at birth), head, band member, content expert, gated by the class or by a random label, with or without a block of its own) and per-name birth |
| `model` | the model assembled from a build: its parts, the velocity, the training loss (over a range of the noise level, with an optional share of whole-range draws) |
| `staged` | a model of stage clusters: one whole model per stretch of the noise level, blended by the stage weights; shared units; staged folders, their links and stale lists |
| `forms` | builds of the stage forms (a cluster: anchor, stem, body, head; one model with mixed tails), adopting units from a build folder, class experts and band members added to a finished build |
| `score` | the picture score: the Frechet distance of a judge's features of generated and training pictures (pictures sampled with Euler steps, or on a time grid with `grid=`), with its mean part and its spread part; a class reader (a closed-form ridge map from the judge's features to the classes) and its check on the test pictures |
| `closed` | heads solved by least squares on a frozen, finished model (normal equations over the training stream, in float64); one head per stage by a solve, written as a staged folder; added parts with no block (class heads) solved jointly |
| `determinism` | the `det=` switch of every experiment: `warn` names the operations that are not deterministic on the device, `on` takes their deterministic algorithms (on a card the attention kernel's backward pass is not deterministic by default, so two runs of one job can differ in their last digits). Measured cost of `on`: none on an RTX 4090, 16% to 20% of the training time on an RTX PRO 6000 |
| `build` | the build file, the unit card, content hashes, unit files (safetensors + card JSON) |
| `linker` | link a build folder (every file checked against its hash), snap units in and out, the stale list; convert a model or a flat weight file into a build folder |
| `cache` | the feature cache: a frozen model's states over a training stream, so heads and members train without the stem; the stored steps can be passed more than once |
| `beds`, `data` | Gaussian fields and CIFAR-style pictures (the training stream can leave classes out or keep only some); CIFAR-10 from parquet to npz and the npz's content hash |
| `train` | the recipe, the loop and the entry point (`python -m geolip.modular_diffusion.train key=value ...`); `fit_units` trains named units of any model, can keep averages of the trained weights beside the last weights (`averages=`: a decay, a power-law average `g8`, a tail average `tail100`; `use_average`), and can continue another run exactly (its optimizer state, its stream position, its average: `start_step`, `optimizer_state`, `average_start`; `write_state`, `read_state`) |
| `run` | the entry point of the experiments (`python -m geolip.modular_diffusion.run <experiment> key=value ...`) |
| `experiments.time_split` | the time split: stage clusters trained alone, together, from copies, against one model over the whole range; `load_job` opens a finished job as a model |
| `experiments.chain_reads` | reads of saved time-split jobs assembled in new ways, with no training: a stage swapped in from another job, clusters at the wrong stages, another crossfade width, sampling on a time grid |
| `experiments.content` | content as an element: the class read by the body at every noise level, only in the first stage, or never; the per-class closed-form anchor; class experts added to a finished model; a class added to a model that never saw it |
| `experiments.closed_forms` | parts that need no training or no card: a head solved by least squares, one head per stage by a solve, class heads added by a solve, band members trained through the live body and from stored states |
| `experiments.stages` | how many stages and where the cuts go: copies of a shared model that continue its run on one stage each, tails trained alone on a frozen stem, a model grown one block at a time |
| `experiments.load` | a CPU load beside another job (for measuring whether a busy machine changes a job's numbers) |
| `read`, `sample` | the read at fixed noise levels (by band, by part, by class); Euler sampling, sampling on a time grid (`euler_grid`, `stage_grid`, `share_counts`) and a sample sheet |
| `report` | markdown tables from result files |
| `jobs` | a roster runner: one job at a time, a stop file, DONE / FAILED markers, a start guard, a wall cap; claims for several runners on one roster, a hook after each job, jobs that need others |
| `hub` | upload a folder to the Hugging Face Hub with retries (token from `HF_TOKEN` only) |
| `environment` | the installed versions against `requirements.txt` |
| `compat` | the flat argument set and the single-file weight layout of the original research script |

## Notebooks

| notebook | what it does | open |
|---|---|---|
| `colab/base.ipynb` | install at a tag, a 50-step smoke run, one 6,000-step base run on CIFAR-10 | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.1.0/colab/base.ipynb) |
| `colab/time_split.ipynb` | the time split on CIFAR-10: the data and its content hash, a smoke run with the picture score, the roster run by several runners, the summary | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.3.1/colab/time_split.ipynb) |
| `colab/elements.ipynb` | content and closed forms on CIFAR-10: the finished time-split runs it starts from, a smoke run with the picture score and the class reader checked, the roster, the summary | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.4.0/colab/elements.ipynb) |
| `colab/stages.ipynb` | stages, growth and content on averaged weights on CIFAR-10: a check of whether a job repeats beside a busy CPU, then the job list with deterministic algorithms on | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.5.0/colab/stages.ipynb) |

## Install

In a hosted notebook (keeps the notebook's own torch):

```
pip install "modular-diffusion @ git+https://github.com/AbstractEyes/modular-diffusion"
```

Extras: `data` (parquet to npz), `hub` (upload to the Hub), `judge` (the picture score: torch-fidelity, torchmetrics),
`guard` (the roster runner's memory guard), `test`; for example `"modular-diffusion[data,hub,judge] @ git+..."`.

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
