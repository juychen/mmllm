# mmllm

Experimental code for small-scale multimodal sequence modeling with Hyena-style architectures on mouse epigenomic tracks.

The repository currently contains two main lines of work:

1. A toy Hyena/CrossHyena regression experiment that predicts 5hmC from 5mC, DNA sequence, and ATAC signal over DMR-centered regions.
2. A prototype `MultiModalStripedHyena` model that extends a `StripedHyena` backbone with additive ATAC and methylation feature encoders.

This is an active research workspace rather than a packaged library. Paths, scripts, and notebooks reflect local experimentation.

## Project status

Current focus is `EXP-0`: a toy experiment using the same 10k regions as the original Hyena toy setup, testing whether Hyena-style layers can learn the relationship between correlated but non-identical 5mC and 5hmC tracks.

Observed results recorded in this repo:

- 2026-04-14: `CrossHyena` with `long_mixer="conv"` and `filter_len=4`, followed by a Hyena layer, reached about `R^2 ~= 0.4` on a 3k-region dataset.
- 2026-04-15: running from 1k to 70k regions reached about `R^2 ~= 0.5` and Pearson `r ~= 0.7` at 70k regions. The model still appears underfit.

## Repository contents

### Training and model code

- `run_multimodal_track_experiments.py`: unified experiment script for arbitrary track-to-track prediction with configurable signal input, context modalities, and target.
- `run_sample_size_experiments.py`: legacy sample-size experiment entry point for the fixed `5mC + sequence + ATAC -> 5hmC` setup.
- `run_atac_query_sequence_context_experiments.py`: legacy experiment entry point for the fixed `ATAC + sequence -> 5hmC` setup.
- `test_hyena.py`: prototype `MultiModalStripedHyena` implementation and toy forward-pass smoke test.
- `test_run.sh`: local launcher for the sample-size experiment sweep.

### Preprocessing helpers

- `merge_meth.sh`: merges replicate methylation bigWig tracks into bedGraph outputs and tabix-indexes them.
- `merge_meth_tobw.sh`: converts merged bedGraph files to bigWig using `bedGraphToBigWig`.
- `merge_m.sh`: example invocation helper.

### Notebooks

- `exp0-initialtest.ipynb`
- `hyena_toy.ipynb`
- `test_crosshyena.ipynb`
- `test-evo2.ipynb`
- `test-nt3.ipynb`
- `19dmratac_test.ipynb`
- `tissue_ontology_mapping.ipynb`

These notebooks contain exploratory analysis and model iteration history.

## Main experiment

`run_multimodal_track_experiments.py` builds a small regression model with:

- query input: any one of `5mc`, `5hmc`, or `atac`
- context input: any non-empty combination of `sequence`, `5mc`, `5hmc`, and `atac`
- target: any one of `5mc`, `5hmc`, or `atac`
- loss: masked MSE over a configurable position mask
- split strategy: non-overlapping genomic region groups, then train/validation split by group

For each requested sample size, the script:

1. Loads DMR metadata.
2. Expands short regions to a fixed target length.
3. Extracts DNA sequence from a genome FASTA.
4. Loads 5mC and 5hmC bedGraph.gz tracks with tabix.
5. Loads ATAC signal from bigWig.
6. Trains `MinimalCrossHyenaRegressor`.
7. Reports validation loss, `R^2`, and Pearson correlation.
8. Writes aggregated results to CSV and JSON.

## Dependencies

There is no pinned environment file in the repo yet. The Python scripts currently depend on:

- Python 3.10+
- `numpy`
- `pandas`
- `torch`
- `pyBigWig`
- `pyfaidx`
- `pysam`

The prototype multimodal model in `test_hyena.py` also depends on the external `vortex` package providing `StripedHyena` and utility code.

The preprocessing shell scripts expect command-line tools including:

- `wiggletools`
- `bgzip`
- `tabix`
- `bedGraphToBigWig`
- `curl`

## Data requirements

The current scripts assume locally available genomics files, including:

- a DMR CSV with columns such as `chr`, `start`, `end`, `length`, and `center`
- a reference genome FASTA
- gzipped and tabix-indexed 5mC and 5hmC bedGraph tracks
- an ATAC bigWig track

Default paths in the training script point to the author's local filesystem, so you will almost certainly need to override them.

Relevant defaults in `run_multimodal_track_experiments.py`:

- `--dmr-csv output/dmr_with_sequences.csv`
- `--genome-fasta /data2st1/junyi/ref/GRCm38.p6.genome.fa`
- `--m5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz`
- `--hm5c-bedgraph /data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz`
- `--atac-bw /data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw`

## Running the unified multimodal sweep

Example:

```bash
python run_multimodal_track_experiments.py \
	--sample-sizes 1000 5000 10000 30000 70000 \
	--input-modality 5mc \
	--context-modalities sequence atac \
	--target-modality 5hmc \
	--dmr-csv output/dmr_with_sequences.csv \
	--genome-fasta /path/to/GRCm38.p6.genome.fa \
	--m5c-bedgraph /path/to/MC_AMY.CG.m.bedGraph.gz \
	--hm5c-bedgraph /path/to/MC_AMY.CG.h.bedGraph.gz \
	--atac-bw /path/to/AMY_MC_track.bw \
	--scheduler cosine \
	--num-epochs 100 \
	--batch-size 64 \
	--output-csv output/multimodal_track_results.csv \
	--output-json output/multimodal_track_results.json
```

This reproduces the old `5mC + sequence + ATAC -> 5hmC` experiment, but the new script also lets you run combinations such as:

- `--input-modality atac --context-modalities sequence --target-modality 5hmc`
- `--input-modality 5hmc --context-modalities sequence 5mc atac --target-modality atac`
- `--input-modality 5mc --context-modalities sequence 5hmc --target-modality 5mc`

Useful arguments:

- `--sample-sizes`: one or more dataset sizes to evaluate
- `--input-modality`: one signal track used as the query branch
- `--context-modalities`: one or more context modalities concatenated along the channel axis
- `--target-modality`: prediction target track
- `--mask-mode`: `cpg_both`, `cpg_forward`, or `all`
- `--target-length`: fixed region length, default `1024`
- `--train-ratio`: fraction of non-overlap groups used for training, default `0.8`
- `--hidden-dim`: hidden width of the regressor, default `64`
- `--num-epochs`: training epochs, default `30`
- `--patience`: early stopping patience, default `5`
- `--scheduler`: `none`, `cosine`, or `plateau`
- `--atac-scaling`: `none` or `minmax`

The script prints per-epoch metrics and saves one summary row per sample size.

## Legacy fixed experiments

The older scripts remain in the repo for convenience, but they are now thin legacy entry points for specific hard-coded modality combinations:

- `run_sample_size_experiments.py`: `5mC + sequence + ATAC -> 5hmC`
- `run_atac_query_sequence_context_experiments.py`: `ATAC + sequence -> 5hmC`

## Local launcher script

`test_run.sh` is a convenience wrapper that:

- activates a local conda environment named `evo2`
- runs the sample-size sweep
- writes timestamped CSV and JSON results into `output/`

It is tailored to one machine and should be treated as an example, not a portable entry point.

## Preprocessing workflow

The methylation preprocessing scripts are intended for local track preparation:

1. `merge_meth.sh` averages replicate bigWig tracks into merged bedGraph files for each condition, region, context, and modification type.
2. The same script compresses outputs with `bgzip` and indexes them with `tabix`.
3. `merge_meth_tobw.sh` converts merged bedGraph files into bigWig using `mm10.chrom.sizes`.

These scripts currently hardcode local input and output directories.

## Prototype multimodal StripedHyena

`test_hyena.py` contains a standalone prototype showing how to extend `StripedHyena` with per-position:

- ATAC-seq features
- methylation features

The extra modalities are encoded with small MLPs and added to token embeddings before the Hyena backbone. The script includes a toy config and a forward-pass smoke test on random DNA plus random feature tensors.

This file is useful as a design sketch, but it is not yet wired into the main experiment pipeline.

## Notes and limitations

- This repo is research code and not yet packaged for reproducible installation.
- Several scripts rely on absolute local paths.
- There is no committed environment file or lockfile yet.
- The `output/` directory is expected by some scripts but is not included in the repo.
- Notebook contents likely contain the most detailed iteration history.

## Next cleanup steps

Practical improvements that would make this repo easier to rerun:

1. Add a `requirements.txt` or `environment.yml`.
2. Replace hardcoded paths with a config file or CLI-only inputs.
3. Add a small example DMR CSV schema and expected track formats.
4. Save representative output tables or plots under version control.
# mmllm
mmllm