# EY AI Model Provenance Kit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![EY AI](https://img.shields.io/badge/EY-AI%20Model%20Provenance-ffe600)](https://github.com/mayanklau/EY-AI-Model-Provenance-Kit-)

EY AI Model Provenance Kit is a Python toolkit and CLI for identifying whether a machine learning model is derived from a known base model family. It compares multi-signal fingerprints extracted from model weights, tokenizers, and architecture metadata to support model governance, supply-chain review, audit readiness, and AI risk workflows.

![Model Provenance Kit Demo](images/demo.gif)

## Key Features

- **Pairwise comparison**: compare two models head-to-head across 8 provenance signals.
- **Database scan**: scan a model against a bundled reference database of known base-model fingerprints.
- **Deep-signal fingerprints**: download pre-computed weight fingerprints for weight-level matching.
- **Multi-signal pipeline**: combine metadata (MFI), tokenizer (TFV, VOA), and weight signals (EAS, NLF, LEP, END, WVC) into one provenance score.
- **MFI gate**: use architecture metadata as a fast structural gate before heavier weight analysis.
- **Two-layer caching**: use in-memory and disk JSON caches for repeat scans.
- **Multiple output formats**: use Rich terminal tables, JSON, or plain text.
- **Streaming support**: load models over 20 GB through streaming to limit memory usage.

## Reference Database

The bundled reference database contains fingerprints for approximately 150 base models spanning 45+ model families and 20+ publishers, ranging from 135M to 70B+ parameters.

The database covers text generation, fill-mask, text-to-text, embedding, and translation architectures across four size buckets: `<=1B`, `1B-10B`, `10B-40B`, and `40B+`.

## Documentation

For deeper technical details, see the guides in [`docs/`](docs/):

| Guide | Description |
|-------|-------------|
| [Pipeline Architecture](docs/architecture.md) | End-to-end data flow, compare vs scan modes, phase breakdown |
| [Signal Reference](docs/signals.md) | Extraction, similarity, and behavior of all 8 provenance signals |
| [Scoring and Model Loading](docs/scoring-and-model-loading.md) | Identity/tokenizer scores, MFI gate, NaN handling, large-model streaming |
| [Database and Caching](docs/database-and-caching.md) | Seed database layout, deep-signal download, two-layer cache, HMAC integrity |

For the formal definition of model provenance, see the [Model Provenance Constitution](docs/constitution/model_provenance_constitution.md).

## Installation

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) recommended, or `pip`

### Install from source

```bash
git clone https://github.com/mayanklau/EY-AI-Model-Provenance-Kit-.git
cd EY-AI-Model-Provenance-Kit-
uv sync
```

### Install as a CLI tool

```bash
uv tool install .
```

After installation, the `provenancekit` command is available:

```bash
provenancekit --help
```

## Quick Start

### 1. Download deep-signal fingerprints

Deep-signal fingerprints are pre-computed weight-level features stored as parquet files. They enable full weight-signal matching during `scan`. Without them, scan results rely on metadata and tokenizer signals.

The default deep-signal archive is hosted as a GitHub release asset:

```bash
https://github.com/mayanklau/EY-AI-Model-Governace-Toolkit-Dataset-/releases/download/deep-signals-v1/deep-signals.zip
```

Then run:

```bash
provenancekit download-deepsignals-fingerprint
```

Check installation status:

```bash
provenancekit download-deepsignals-fingerprint --status
```

Update fingerprints:

```bash
provenancekit download-deepsignals-fingerprint --update
```

### 2. Scan a model against known base models

```bash
provenancekit scan bigscience/bloom-560m
```

This extracts features from the model, runs a 3-stage lookup against the reference database, and returns ranked matches with scores and decision labels.

### 3. Compare two models head-to-head

```bash
provenancekit compare gpt2 distilgpt2
```

## Usage

### Commands

| Command | Purpose |
|---------|---------|
| `provenancekit compare MODEL_A MODEL_B` | Pairwise comparison of two models |
| `provenancekit scan MODEL_ID` | Scan one model against the reference database |
| `provenancekit download-deepsignals-fingerprint` | Download/manage deep-signal weight fingerprints |

### Output Formats

```bash
# Rich terminal table
provenancekit compare gpt2 gpt2

# JSON for automation
provenancekit compare gpt2 gpt2 --json

# Plain text for CI logs
provenancekit compare gpt2 gpt2 --plain
```

### Verbose Logging

Enable structured logging to stderr with the top-level `--verbose` flag:

```bash
provenancekit --verbose scan bigscience/bloom-560m
provenancekit --verbose compare gpt2 distilgpt2
```

## CLI Reference

### `provenancekit compare`

```text
provenancekit [--verbose] compare MODEL_A MODEL_B [options]
```

| Option | Description |
|--------|-------------|
| `MODEL_A` | First model: Hugging Face repo ID or local path |
| `MODEL_B` | Second model: Hugging Face repo ID or local path |
| `--json` | Output as JSON |
| `--plain` | Output as plain key-value text |
| `--cache-dir PATH` | Override the default cache directory |
| `--no-cache` | Disable feature caching |
| `--timing` | Show high-level phase timings |

### `provenancekit scan`

```text
provenancekit [--verbose] scan MODEL_ID [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_ID` | | Model to scan: Hugging Face repo ID or local snapshot path |
| `--json` | | Output as JSON |
| `--plain` | | Output as plain key-value text |
| `--top-k N` | `3` | Maximum number of matches to return |
| `--threshold F` | `0.50` | Minimum pipeline score for inclusion |
| `--db-root PATH` | bundled DB | Override the provenance database root directory |
| `--cache-dir PATH` | `~/.provenancekit/cache` | Override the default cache directory |
| `--no-cache` | | Disable feature caching |
| `--timing` | | Show phase-level timing breakdown |

### `provenancekit download-deepsignals-fingerprint`

```text
provenancekit download-deepsignals-fingerprint [options]
```

| Option | Description |
|--------|-------------|
| `--db-root PATH` | Override the provenance database root directory |
| `--update` | Re-download and replace existing fingerprints |
| `--no-verify` | Skip SHA-256 integrity check after download |
| `--status` | Show current deep-signals installation status and exit |

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

The package entry point is `provenancekit`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
