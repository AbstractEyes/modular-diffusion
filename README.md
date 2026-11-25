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
| `layout`, `experiments.layout_write` | written picture layouts: prompts with known answers, the box, cell and chain forms with their strict readers, the address of a thing instance on a 3 x 3 grid, the scores and the controls; the experiment that asks a writer (a fake, the text parser, or a chat model with an optional LoRA merged in) and scores it (see below) |
| `scenes`, `scene_reads`, `scene_judge`, `collective`, `experiments.toy_collective` | synthetic scenes of a 3 x 3 grid with known layouts; the reads of a made scene; the cell judge; engines that read the address frame (two read forms), their trainer and the collective's joins; the experiment (see below) |
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
| `capture.loss`, `capture.train` | the capture loss and one band engine trained alone (pure Adam, an averaged copy, exact resume from a checkpoint); from the teacher's numbers or from random ones (`init`); reads during training by a hook that changes no number of the training and survives a resume (`every`, `on_every`); a finished job may drop its checkpoint and is then read back, never trained again (`keep_checkpoint`); the median seconds a step and the trained parameter count in train.json |
| `capture.assemble` | the assembly (parts blended by the band weights; only live parts run) and a deterministic DDIM sampler with guidance |
| `capture.reads` | probes (float32 latents and noise; their rows read all at once or in batches) and reads: a band's read, the born read, the exchange test (a part put back for itself reads exactly 0), the off-band control; a probe sampled in groups of rows, group j from the noise of seed + 100003 j, the same for every model compared (`sample_probe`); the chain read (`chain_read`): a model's guided answers against the teacher's on the states of the sampler's chain, on the model's own chain and on the teacher's, per band the mean error on each and their ratio with its standard error |
| `capture.measure`, `capture.data` | the teacher measured per band with no training (the born read of every single removal; how much each band needs the prompt; measured cost); latent shards and batch sources; rows of the shards in file order (`rows_in_order`); parquet pictures after a skip, any caption of a list (`pick`) |
| `capture.pictures` | an assembly's final latents sampled group by group with the probe's start-noise rule (`sample_latents`), latents decoded to pictures (`decode`), the kernel distance with its blocks (`kid`), a text-to-picture judge on a local CLIP model and a fake one for tests, a contact sheet |
| `experiments.capture` | `python -m geolip.modular_diffusion.run capture variant=<band_engine, born, assembly, exchange, pictures, picture_compare or chain_read> ...` (`tiny=1` for a smoke run on the tiny random teacher); a probe of held-out rows in file order (`probe_data`, `probe_first`, `probe_rows`, `chunk`); for band_engine `init`, `read_every` (the read curve) and `keep_checkpoint`; for exchange `parts`, `band`, `other`; sampling `group` rows at a time |
| `hub` | `ship(folder, repo, path, message, ignore=("*.safetensors", "checkpoint.pt"))` uploads a folder without the files the patterns name (a pattern without a slash matches at any depth) |

An engine's numbers are float32; the teacher may be kept in half precision on a card (every part casts its inputs to
its own precision and answers in the latents'). `tests/test_capture_cuda.py` runs the reads, the measurements, the
sampler and the trainer with its resume on a card, and the random start, the reads during training, a held-out probe
and the exchange variant (off unless `MD_TEST_CUDA=1`).

A part in `parts=` (and `other=`) is `teacher`, a band engine's job folder, or an engine born from the teacher and not
trained: `born:<shape>` (it reads the prompt) or `born:<shape>:notext` (it does not), for example `born:2222/nomid/3`.
The `pictures` variant judges an assembly by its pictures rather than by its distance from the teacher: for rows of a
folder of latent shards in file order (their captions are the prompts), it samples the assembly's pictures from a fixed
start noise, group by group, decodes them with the teacher's image decoder (kept in full precision), and saves a
judge's features of every picture, optionally a text-to-picture score from a local CLIP model, and a contact sheet;
`source=rows` decodes the rows' own pictures instead, the reference set, and `part_dtype` runs the engines in half
precision. The `picture_compare` variant reads two such job folders and gives the Frechet distance and the kernel
distance between their pictures and, row by row, the difference of their text-to-picture scores; two jobs whose rows
are the same, or one a prefix of the other, sampled with the same group size, noise seed, steps and guidance, start
every common row from the same noise.

## Layouts: how often a writer writes the right addresses

`geolip.modular_diffusion.layout` (the standard library only) measures a writer of picture layouts from a text prompt:
which things, how many, which color on which, and where each goes on a 3 x 3 grid. Each thing instance gets one
ADDRESS, a cell of the grid (from a box on the 0 to 1000 grid: the cell holding its center, cut at 0.38 and 0.62 of
the width and the height; from words: the named cell). 600 prompts have answers known by construction (100 each of
count, absent, colour, place, relation and combined, from a seeded generator), three forms of written answer are read
by strict grammars (a JSON list of boxes, one line a thing `name | color | column | row`, a written chain of thought
read line by line), and every prompt is scored 0 or 1 on parsed, things, counts, colors, placed, spatial, no conflict
(no cell holds two different things) and absent, with ADDRESS RIGHT the product of the scores of its family; the
published rules of three layout papers are computed beside it on the raw boxes. Controls: the generator's own layouts
score 1, answers scored against another prompt of the family score near 0, broken answers each give their own kind of
failure, and a repeated ask gives the same text.

```
address of a box: col = left if c_x < 0.38, right if c_x > 0.62, else centre (rows the same on c_y)
rate p over n distinct prompt texts with standard error sqrt(p (1 - p) / n) (a text asked several times counts once
when its copies are scored alike); two writers compared text by text:
d = mean(x_a - x_b), se_d = sd(x_a - x_b) / sqrt(n), a tie when |d| < 2 se_d
```

`python -m geolip.modular_diffusion.run layout_write variant=write writer=<fake:truth | parser | model> form=<box | cell
| chain_records | chain_text | spec> examples=1 out=<folder>` runs one cell (extra `layout` for a model writer;
`writer=parser` and `form=chain_records` need the `thought-image-seeds` and `qwen-symbolic-thinking` packages);
`variant=write_compare a=<cell> b=<cell>` compares two cells prompt by prompt. `examples=2` puts nine worked examples
in the ask (the six, one of each family, followed by three of the combined kind); `examples=0` none. A cell-form answer
is also read by the table-tolerant read (`layout.read_table`: a Markdown table's code fence, outer bars, header line and
separator line taken off, then the strict grammar), whose scores every row and summary carry beside the strict ones.

## The address frame on image tokens and the toy collective

`scenes` draws synthetic scenes with exactly known layouts: a 48 x 48 canvas cut into a 3 x 3 grid of 16-pixel cells,
each holding one of nine things (three families of three) in one of nine colors, or nothing, on a flat gray ground
(bed F) or on one tint a scene with a smooth texture across the cells (bed C). `frame` (its last section) gives every
image token (patch 4: a 12 x 12 grid) its place digits (column, row, place inside the cell) and every request (one
thing in one cell) six digits of base 3, each digit coded on orthonormal rotary anchors; the gate is declared:

```
w_i(token)    = sum over requests r of <u(j), u(j_r)> <u(k), u(k_r)> [thing_r in T_i]
w_base(token) = 1 - sum over requests r of <u(j), u(j_r)> <u(k), u(k_r)>
```

one product a request, summed over requests, so every token has exactly one owner. `collective` builds small engines
(width 128, 2 heads, 3 blocks, a zero-born head) that read the frame in one of two forms: the address codes added to
the token embeddings with the request in the cross-attention context (`read=embedding`), or the request as a token of
the sequence placed by an anchored rotary whose pairs turn with the token's row and column, stay still, or turn by the
cell's column and row digits (`read=rotary`). A part trained alone on one-thing scenes sees only its own cell and the
cells no other request owns (a key mask in the block's self-attention) and is trained on its own cell's tokens; the
collective serves a picture by selection, token by token (`gate`), or by the published sum of answers over an empty or
an unconditional background. `scene_reads` scores a made scene cell by cell with a small judge (`scene_judge`), and
reads its tint coherence, its seams and its stray ink. `python -m geolip.modular_diffusion.run toy_collective
variant=<parts | judge | timing | compose | controls | exact | leak | wrong> ...` trains the parts and reads the rows.

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
| `colab/first_capture.ipynb` | the first capture: the clean band's engine of Stable Diffusion 1.5, with no prompt input, made three ways on the shape `tiny` (the model's weights inherited and taught by the model; random weights taught by the model; random weights taught by the noise) and inherited on `2222/nomid/3`, two seeds each, read every 250 steps on 64 held-out rows; then served in the sampler beside the model, exchanged for its twin and served off its band; a smoke run of every job kind, the sampler checked against the diffusers DDIM scheduler, a timing; a copy of the model's license and a notice shipped first, then each job folder with its engine files; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.9.1/colab/first_capture.ipynb) |
| `colab/three_band.ipynb` | the three-band assembly: the middle and the noisy bands' engines of Stable Diffusion 1.5, inherited and reading the prompt, on the shapes `tiny` and `2222/nomid/3`, two seeds each, and one `tiny` engine trained on the whole noise range at the same total steps; the distance reads of the first capture (each engine beside the model, the three-band assembly with the first capture's clean engines, exchanges, each engine served in the other band); then every assembly judged by its pictures on 1,000 held-out prompts: Inception features, the kernel and Frechet distances against the model's own pictures and the real ones, the CLIP text-to-picture score, the model against itself from another start noise as the floor; a smoke run of every job kind with its exactness checks, the real judges checked on 16 pictures, a timing; the license copy and notice first, then each job folder; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.10.0/colab/three_band.ipynb) |
| `colab/three_band_part2.ipynb` | the three-band assembly, part 2: whether inheriting the model's weights is what makes the bands that read the prompt work: the shape `tiny` from random weights, taught by the model's answers and reading the prompt, on the middle and the noisy band, two seeds each, by part 1's recipe and data; each engine beside the model in the sampler, each band's first engine judged by its pictures on 500 held-out prompts against the model's pictures and the real ones; the model's own pictures brought back from part 1's results; a smoke run of every job kind, the real judges checked on 16 pictures, a timing; each job folder shipped as it finishes; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.11.0/colab/three_band_part2.ipynb) |
| `colab/three_band_pairs.ipynb` | the three-band assembly, where it loses: sampling only, nothing trained, and it runs on a 15 GiB card; the band engines of the three-band notebook served two at a time (the model itself in the third band) on 500 held-out prompts, and an assembly that mixes the two shapes on 1,000, each judged by its pictures against the model's and the real ones; a parity check of this card's pictures of the model against the earlier ones; the chain read of each three-band assembly, of the whole-range engine and of an assembly with an untrained noisy part (how far an assembly is from the model on its own sampling chain and on the model's); the finished engines and reference pictures brought back from the Hub (the cell stops if one is missing); a smoke run with one chain read on the tiny model, the real judges checked on 16 pictures, one chain read of the model timed for the projection; each job folder shipped as it finishes; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.14.0/colab/three_band_pairs.ipynb) |
| `colab/layout_write.ipynb` | who writes the addresses of a picture and how often they are wrong: the text parser alone, Qwen3.5-2B and its LoRA fine-tune (merged) writing layouts for 600 prompts with known answers and 100 real COCO captions, as boxes, as grid cells and in the fine-tune's own text-only forms (the short form and the whole row with its question lines), with and without six worked examples (13 cells); a smoke run of every cell with the scorer's controls and a projection; each cell shipped as it finishes; the prompt-by-prompt comparisons; the machine kept until everything finished is shipped; the first cell removes a torchao older than 0.16 when the runtime ships one (PEFT cannot build a LoRA beside it) | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.11.1/colab/layout_write.ipynb) |
| `colab/layout_sizes.ipynb` | the layout writer by model size and by worked examples: Qwen3.5-2B, -4B and -9B and the LoRA fine-tune of each (merged), writing layouts in the grid-cell form for the same 600 prompts with known answers and 100 real COCO captions, with six worked examples, with nine (three more of the composite kind) and with none (16 cells), each answer read by the strict grammar and by the table-tolerant read; the prompt-by-prompt comparisons inside a size (nine against six examples, the fine-tune against its base); a `SIZES` choice (all, 2b+4b, 9b) so that two machines can each run a part; a smoke run of every chosen cell with the scorer's controls, the 9B batch settled by the card's memory, and a projection; each job shipped as it finishes; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.13.0/colab/layout_sizes.ipynb) |
| `colab/toy_collective.ipynb` | small engines that each learned one thing alone, served by place and thing addresses on the image tokens of synthetic scenes: the judge of each bed trained first with its ceiling; a smoke run of every job kind; the seconds a step with 1, 2 and 4 trainings at once (the runner count is the best throughput); then the parts of both read forms, two seeds, the whole models, and every row (the collective, the whole-picture parts by the gate and by the published sums, the whole models, real scenes) judged on 500 request lists a size, with the controls, the exactness reads, the leak, the wrong-address read, the dose and an address never trained; several runners over one roster ordered so that each row is read as soon as the parts it reads exist; each job folder shipped as it finishes; the largest model's step count fixed by the run's first machine and reused on any later one; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.12.1/colab/toy_collective.ipynb) |
| `colab/toy_prompt_to_picture.ipynb` | from a caption to a picture at toy size, nothing trained: 600 captions of the synthetic scenes in six families (counts, an absent thing, colours, places, relations, several things with their counts, colours and places), each with its true layout; a language model writes a layout on a 3 x 3 grid for each, the layout is handed to the toy collective as requests (one thing a cell), the collective of the earlier notebook's engines (both seeds, fetched from the Hub; the cell stops if one is missing) draws it and its judge reads every cell; per caption whether the layout was written right, handed on right, drawn as written and whether the picture satisfies the caption; two writers (a chosen one and a small one) beside a writer of the true layout and one of another caption's layout; a parity check of the collective against the earlier run's own numbers; a smoke run of both writers through the whole chain and the projected minutes; each job folder shipped as it finishes; the machine kept until everything finished is shipped | [Open in Colab](https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/v0.15.0/colab/toy_prompt_to_picture.ipynb) |

## Install

In a hosted notebook (keeps the notebook's own torch):

```
pip install "modular-diffusion @ git+https://github.com/AbstractEyes/modular-diffusion"
```

Extras: `data` (parquet to npz), `hub` (upload to the Hub), `judge` (the picture score: torch-fidelity, torchmetrics),
`guard` (the roster runner's memory guard), `capture` (diffusers and transformers, for `capture`), `layout`
(transformers, accelerate, peft, for a model writing layouts), `test`; for example `"modular-diffusion[data,hub,judge] @ git+..."`.

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
