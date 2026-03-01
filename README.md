# PRISM

Code for **PRISM: PRIor from corpus Statistics for topicModeling**

**TL;DR:** This repo requires a patched **MALLET 2.0.8** source build to support **β as a vector (per vocabulary type)** rather than a scalar.

This repository provides the full implementation used in the paper, including (i) construction of a corpus-intrinsic structure signal from text, and (ii) injection of that signal into an LDA-style topic model through a **vector-valued β prior** (per vocabulary type), enabling fine-grained, interpretable control over topic–word distributions beyond what a scalar β can express.

<p align="center">
  <img src="figures/PRISM_Framework.png" alt="Pipeline overview" width="900"/>
</p>
<p align="center">
  <em>Figure: End-to-end pipeline implemented in this repository.</em>
</p>

---

## Why this matters
Topic models are often used as “downstream summarizers” of a corpus, but their most critical modeling choice—the topic–word prior—tends to be treated as a single global scalar (β). That simplification limits what the model can express: it cannot encode **word-specific inductive bias** derived from the corpus itself (e.g., systematic structure, co-occurrence geometry, or graph-derived signals).

This project addresses that gap by enabling a **β vector over vocabulary types**, which supports:
- **Per-word regularization**: different words can be encouraged or discouraged with different strengths.
- **Structure-aware priors**: corpus-intrinsic signals (e.g., graph-derived weights) can be injected in a principled way.
- **More controllable topic formation**: topics can become both sharper and more semantically coherent when the prior reflects intrinsic corpus structure.
- **Interpretability**: the prior is explicit and inspectable, making it easier to understand *why* certain words dominate particular topics.

In short: this repo is for practitioners and researchers who want topic models whose inductive bias is *learned from the corpus itself* and then enforced transparently through the generative model.

---

## Quick Start: Install MALLET + Apply Patch

### 1) Clone this repo
```bash
git clone https://github.com/tal-ishon/PRISM.git
cd PRISM
```

### 2) Run the setup script (downloads + patches + builds MALLET)
```bash
chmod +x setup_mallet.sh
./setup_mallet.sh
```
What this does:

- Downloads MALLET 2.0.8 source

- Extracts it and renames the folder to `./mallet`

- Replaces `./mallet/src/cc/mallet/` with the patched directory in this repo (`beta_vector_patch/mallet`)

- Builds MALLET with `ant clean && ant`

### 3) Verify it worked
You should now have:

- Patched source at: `./mallet/src/cc/mallet/`
- Backup of original MALLET code at: `./mallet/src/cc/mallet.ORIG/`

Optional sanity check:
```bash
cd mallet
./bin/mallet --help
```
### Troubleshooting (common)

`ant: command not found` → install Apache Ant, then rerun the script.

Java issues → check `java -version` and `javac -version`, ensure a working JDK is installed.
