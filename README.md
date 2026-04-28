# TBEF

This repository contains the implementation of **TBEF** for event trigger extraction on the **MAVEN** dataset.

The method described in the paper can be summarized as:

**TBEF = TAP-BERT + Event Fusion Layer + CRF**

In the current codebase, the **TAP-BERT** component is realized through three implementation modules:

1. **Word embedding decomposition** via gated TSVD-based low-rank embedding compression;
2. **Attention pooling** via a multi-head sentence-level attention pooling module;
3. **Feed-forward neural network pruning** via periodic structured FFN pruning during training.

On top of TAP-BERT, the repository further implements the **event fusion layer** at the sentence level and uses a **linear-chain CRF** for final BIO sequence decoding.

Therefore, the default training pipeline in this repository can be summarized as:

**BERT encoder + gated low-rank embedding + multi-head attention pooling + sentence-level event fusion + CRF decoding + periodic FFN pruning**

Although MAVEN is organized at the document level, the current implementation performs **sentence-level event trigger detection**: each document is decomposed into sentence instances, and each sentence is treated as an independent token classification sample.

---

## Method–Paper Alignment

The implementation modules correspond to the paper as follows:

- **TAP-BERT layer**
  - `compressed_embedding.py`: word embedding decomposition
  - attention pooling in `bert_crf.py`: self-attention pooling enhancement
  - `--prune_ffn`: FFN pruning

- **Event Fusion Layer**
  - `--use_sentence_event_fusion`: sentence-level implementation of the event fusion layer

- **CRF layer**
  - `crf.py`: linear-chain CRF for BIO sequence decoding

In other words, the code-level options such as gated embedding compression and multi-head attention pooling should be understood as the concrete implementation of the corresponding modules described in the paper.

---

## Features

- BERT encoder with `bert-base-uncased`
- Gated TSVD embedding compression
- Multi-head attention pooling
- Sentence-level event fusion
- CRF sequence decoding
- Periodic structured FFN pruning
- Training-time validation, early stopping, and best-checkpoint saving
- Local MAVEN data directory resolved relative to `run_maven.py`

---

## Project Layout

```text
TBEF/
├── README.md
├── run_maven.py              # CLI entry point for training and evaluation
├── bert_crf.py               # Model, pooling, sentence fusion, CRF bridge, pruning
├── compressed_embedding.py   # Gated TSVD embedding module
├── crf.py                    # Linear-chain CRF
├── utils_maven.py            # MAVEN labels and sentence-level feature conversion
└── maven/
    ├── train.jsonl
    ├── valid.jsonl
    ├── test.jsonl
    ├── cached_train_bert-base-uncased_128
    ├── cached_valid_bert-base-uncased_128
    └── cached_dev_bert-base-uncased_128
````

---

## Requirements

### Verified Python Package Versions

```text
Python 3.10
torch==2.6.0+cu124
transformers==4.57.6
seqeval==1.2.2
numpy==2.2.6
tqdm==4.67.3
```

Install the Python dependencies in an environment with a CUDA-compatible PyTorch build:

```bash
pip install torch transformers seqeval numpy tqdm
```

---

## Data Configuration

TBEF expects MAVEN-format JSONL files. The default data directory is:

```text
./maven
```

`run_maven.py` resolves this path from the script location, so the command can be launched from any working directory. `--data_dir` is optional and should only be provided when intentionally using a different MAVEN-format directory.

### Bundled Data

The bundled `maven/` directory is ready to use:

```text
maven/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

### Recreate the Data Directory

If the dataset directory needs to be recreated, download the official package and place the JSONL files under `maven/`:

```bash
mkdir -p maven

curl -L "https://cloud.tsinghua.edu.cn/d/874e0ad810f34272a03b/files/?p=/train.jsonl&dl=1" -o maven/train.jsonl
curl -L "https://cloud.tsinghua.edu.cn/d/874e0ad810f34272a03b/files/?p=/valid.jsonl&dl=1" -o maven/valid.jsonl
curl -L "https://cloud.tsinghua.edu.cn/d/874e0ad810f34272a03b/files/?p=/test.jsonl&dl=1" -o maven/test.jsonl
```

Expected raw data files:

```text
maven/train.jsonl
maven/valid.jsonl
maven/test.jsonl
```

`train.jsonl` and `valid.jsonl` contain `events` and `negative_triggers`.
`test.jsonl` contains `candidates` because test labels are hidden in MAVEN.

### Feature Caches

The included caches match:

* model name: `bert-base-uncased`
* sequence length: `128`
* modes used by training/evaluation: `train`, `valid`, and `dev`

If you change `--max_seq_length`, tokenizer, or data files, delete the matching `maven/cached_*` file and rerun so features are regenerated.

For the default configuration, keeping the bundled caches avoids preprocessing time. To force cache regeneration:

```bash
rm -f maven/cached_train_bert-base-uncased_128
rm -f maven/cached_valid_bert-base-uncased_128
rm -f maven/cached_dev_bert-base-uncased_128
```

The script will recreate the required cache files on the next run.

### Use a Different Data Directory

To use another MAVEN-format directory, pass `--data_dir`:

```bash
python run_maven.py \
  --data_dir /path/to/maven \
  --model_type bertcompressedcrf \
  --model_name_or_path bert-base-uncased \
  --output_dir outputs/custom_data_run \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_attention_pooling \
  --attention_pooling_variant multihead \
  --use_sentence_event_fusion \
  --compression_method gate
```

The custom directory must contain at least:

```text
train.jsonl
valid.jsonl
```

`test.jsonl` is only needed for workflows that inspect or extend test-time candidate prediction outside the default training/evaluation path.


## Reproducibility Notes

For reproducibility, this repository keeps the following implementation assumptions explicit:

* encoder backbone: `bert-base-uncased`
* sentence-level MAVEN training pipeline
* maximum sequence length: `128`
* gated low-rank embedding with TSVD dimension `256`
* multi-head attention pooling with `4` heads
* sentence-level event fusion enabled
* CRF sequence decoding
* periodic FFN pruning during training

If you change the tokenizer, sequence length, data directory, or model switches, please regenerate the feature cache and record the modified hyperparameters in the corresponding experiment directory.

For reproducibility and reference, the supplementary appendix PDF mentioned in the paper is provided in this repository under the filename `TBEF_appendix.pdf`.

## Citation

If you use this repository, please cite the corresponding TBEF paper and clearly state the exact configuration used in your experiments.

