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
| `frame` | the address frame: every part's gate written once as one projective gate over declared addresses. A row's stage digit turns at constant angular speed between two stage anchors across a crossfade; a part's factor is the squared length of the row's address projected on the anchors it owns, and a gate is the product of its digits' factors. The crossfaded window is exactly this (cos^2 / sin^2), a class list is set membership, so the gates of every unit are this gate and no number changes. Also two amplitude forms over the same addresses (plain and discounted) with their roster table, and a check that a model's stage owners cover the noise range exactly once |
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
| `train` | the recipe, the loop and the entry point (`python -m geolip.modular_diffusion.train key=value ...`); `fit_units` trains named units of any model, can keep averages of the trained weights beside the last weights (`averages=`: a decay, a power-law average `g8`, a tail average `tail100`; `use_average`), and can continue another run exactly (its optimizer state, its stream position, its average: `start_step`, `optimizer_state`, `average_start`; `write_state`, `read_state`); a continued state may cover only some of the trainable parameters (`optimizer_partial`: parts born into a running model start without moments), and single rows of a table can be the only trainable part of a parameter (`trainable_rows`: the other rows are restored and checked bit for bit, in the weights and in every average) |
| `run` | the entry point of the experiments (`python -m geolip.modular_diffusion.run <experiment> key=value ...`) |
| `experiments.time_split` | the time split: stage clusters trained alone, together, from copies, against one model over the whole range; `load_job` opens a finished job as a model |
| `experiments.chain_reads` | reads of saved time-split jobs assembled in new ways, with no training: a stage swapped in from another job, clusters at the wrong stages, another crossfade width, sampling on a time grid; a tree of the stages experiment is read the same way from its averaged folder |
| `experiments.content` | content as an element: the class read by the body at every noise level, only in the first stage, or never; the per-class closed-form anchor; class experts added to a finished model; a class added to a model that never saw it; experts born with identity blocks or born by a solve (their heads solved in closed form on the frozen body before any training), per class or per class and stage |
| `experiments.closed_forms` | parts that need no training or no card: a head solved by least squares, one head per stage by a solve, class heads added by a solve (for all classes or for chosen ones), band members trained through the live body and from stored states, band members and one unfiltered member on a finished job's averaged weights |
| `experiments.stages` | how many stages and where the cuts go: copies of a shared model that continue its run on one stage each, tails trained alone on a frozen stem, a model grown one block at a time; the thaw pass: such a build trained on with its top part only, with the base's last one or two blocks, or with everything set trainable; the stage tree and its parts: a shared run that reads no class or leaves classes out, forked at any step into copies (on the source's batches or on a sibling stream), class-gated parts born with the shared run, at the fork, or added afterwards to the finished tree with each stage's parts trained alone on its own range (variant `parts`); a class added to a finished tree by one row of the class table, by a block, or by both; trained parts placed untrained on another tree (`init_from`, a graft: the result records whether the parents are equal), or one stage's parts used by every stage |
| `experiments.load` | a CPU load beside another job (for measuring whether a busy machine changes a job's numbers) |
| `read`, `sample` | the read at fixed noise levels (by band, by part, by class); Euler sampling, sampling on a time grid (`euler_grid`, `stage_grid`, `share_counts`) and a sample sheet |
| `report` | markdown tables from result files |
| `jobs` | a roster runner: one job at a time, a stop file, DONE / FAILED markers, a start guard, a wall cap; claims for several runners on one roster, a hook after each job, jobs that need others |
| `hub` | upload a folder to the Hugging Face Hub with retries (token from `HF_TOKEN` only) |
| `environment` | the installed versions against `requirements.txt` |
| `compat` | the flat argument set and the single-file weight layout of the original research script |

## Capture: band engines from an existing U-Net

`geolip.modular_diffusion.capture` (extra `capture`: diffusers, transformers) takes Stable Diffusion 1.5 apart along
the noise range. The teacher stays frozen and is read from a local diffusers folder (nothing is downloaded).

```
steps t = 0 .. 999 (larger is noisier); z_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps; the teacher predicts eps
band coordinate s = t / 1000; bands clean / middle / noisy with edges 0.35, 0.75 and a crossfade half-width 0.06
w_clean = 1 - up1, w_middle = up1 (1 - up2), w_noisy = up1 up2, up1 = ramp(s - 0.35), up2 = ramp(s - 0.75)
capture loss L = lam_out mean((o_S - sg(o_T))^2) + lam_data mean((o_S - eps)^2)      (lam_out 1, lam_data 0)
```

| module | what it holds |
|---|---|
| `capture.schedule` | the teacher's schedule; the band weights, served by the address frame's projective gate over three declared windows; step sets; the band sampling law with a whole-range share |
| `capture.teacher` | `load_teacher` from a local folder, `find_snapshot` in the local hub cache, and a tiny random teacher for tests |
| `capture.shapes`, `capture.engine` | engine shapes (layers kept per stage, middle block, three or four stages; `full`, `base`, `small`, `tiny`) and engines made by paired block removal, every tensor copied from the teacher exactly; an engine that ignores the prompt; size and measured cost |
| `capture.loss`, `capture.train` | the capture loss and one band engine trained alone (pure Adam, an averaged copy, exact resume from a checkpoint) |
| `capture.assemble` | the assembly (parts blended by the band weights; only live parts run) and a deterministic DDIM sampler with guidance |
| `capture.reads` | probes (float32 latents and noise; their rows read all at once or in batches) and reads: a band's read, the born read, the exchange test, the off-band control |
| `capture.measure`, `capture.data` | the teacher measured per band with no training (the born read of every single removal; how much each band needs the prompt; measured cost); latent shards and batch sources |
| `experiments.capture` | `python -m geolip.modular_diffusion.run capture variant=<band_engine, born or assembly> ...` (`tiny=1` for a smoke run on the tiny random teacher) |

An engine's numbers are float32; the teacher may be kept in half precision on a card (every part casts its inputs to
its own precision and answers in the latents'). `tests/test_capture_cuda.py` runs the reads, the measurements, the
sampler and the trainer with its resume on a card (off unless `MD_TEST_CUDA=1`).

## Notebooks

| notebook | what it does | open |
|---|---|---|
| `colab/base.ipynb` | install at a tag, a 50-step smoke run, one 6,000-step base run on CIFAR-10 | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.1.0/colab/base.ipynb) |
| `colab/time_split.ipynb` | the time split on CIFAR-10: the data and its content hash, a smoke run with the picture score, the roster run by several runners, the summary | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.3.1/colab/time_split.ipynb) |
| `colab/elements.ipynb` | content and closed forms on CIFAR-10: the finished time-split runs it starts from, a smoke run with the picture score and the class reader checked, the roster, the summary | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.4.0/colab/elements.ipynb) |
| `colab/stages.ipynb` | stages, growth and content on averaged weights on CIFAR-10: a check of whether a job repeats beside a busy CPU, then the job list with deterministic algorithms on | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.5.0/colab/stages.ipynb) |
| `colab/parts.ipynb` | the joint pass after a build on a frozen base against the reference at equal compute, content experts born silent or by a solve, a class added by a solve, band members on averaged weights; every job with deterministic algorithms on | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.6.0/colab/parts.ipynb) |
| `colab/tree.ipynb` | the stage tree and its parts, every model trained by the list itself (so the numbers compare on any card type): first a class added to a finished tree by one row of the class table or by a block, each trained alone, what it costs, and what such a part keeps when it is reused untrained (on the tree trained further, on a sibling tree, on another run's tree, as one part for every stage), against retraining the whole tree; then when to fork a shared run into one copy per stage, the tree against the plain model at twice the compute and at equal size, and class-gated parts born with the body, at the fork or afterwards | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.7.1/colab/tree.ipynb) |
| `colab/reuse.ipynb` | how far a block that adds a class to a finished stage tree reaches once it is made: one block trained once for every stage, a block from another run as the starting point of a short re-fit, the block and one row of the class table at several training lengths, and the block placed untrained on trees whose shared run was trained further or forked earlier; every model trained by the list itself | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.7.2/colab/reuse.ipynb) |
| `colab/band_needs.ipynb` | what each noise band of Stable Diffusion 1.5 needs, with no training (`python -m geolip.modular_diffusion.run band_needs ...`): 256 captioned COCO pictures with their content hash, encoded once by the model's own image encoder; a smoke run; a measured projection of the run's minutes; then the born reads of every single block removal and of the shapes base, small and tiny on each band, the prompt's weight per band by sampling, and the measured cost of every shape, on two probe seeds, resumable unit by unit | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.8.0/colab/band_needs.ipynb) |

## Install

In a hosted notebook (keeps the notebook's own torch):

```
pip install "modular-diffusion @ git+https://github.com/AbstractEyes/modular-diffusion"
```

Extras: `data` (parquet to npz), `hub` (upload to the Hub), `judge` (the picture score: torch-fidelity, torchmetrics),
`guard` (the roster runner's memory guard), `capture` (diffusers and transformers, for `capture`), `test`; for example `"modular-diffusion[data,hub,judge] @ git+..."`.

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
