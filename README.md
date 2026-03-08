# PRISM: PRIor from corpus Statistics for topic Modeling (TMLR 2026)

Authors: [Tal Ishon](https://www.linkedin.com/in/tal-ishon/), [Yoav Goldberg](linkedin.com/in/yoav-goldberg-8463011), [Uri Shaham](https://www.linkedin.com/in/urishaham/)

>Topic modeling seeks to uncover latent semantic structure in text, with LDA providing a foundational probabilistic framework. While many recent methods incorporate external knowledge (e.g., pre-trained embeddings), such reliance can limit applicability in emerging or underexplored domains. **PRISM** is a **corpus-intrinsic** method that derives a **Dirichlet parameter** from word co-occurrence statistics to initialize LDA **without altering its generative process**. Experiments on text and single-cell RNA-seq data show that PRISM improves topic coherence and interpretability, often rivaling externally guided approaches.

<div align="center">

<a href="https://shaham-lab.github.io/PRISM/">
  <img src="https://img.shields.io/static/v1?label=Project&message=Website&color=9a031e" height="20.5">
</a> 
<a href="https://arxiv.org/????">
  <img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=fb5607" height="20.5">
</a>

<br>

<em>Topic Modeling • Corpus-intrinsic representation • Graph-derived priors</em>

</div>

## Overview
PRISM is a corpus-intrinsic framework for enhancing LDA by distilling informative structure directly from the target corpus, without reliance on external knowledge, making it particularly well suited to data-scarce and under-characterized domains such as biological scRNA-seq.

**Core goals**

- **Methodological goal (text)**:
Develop PRISM as a state-of-the-art corpus-intrinsic topic modeling framework that improves LDA’s coherence and interpretability using only corpus-derived signals, and remains competitive with approaches that rely on external resources.
- **Scientific goal (biology)**:
Translate PRISM to scRNA-seq to discover gene-expression programs in data-scarce, under-characterized settings, where curated priors and pretrained representations may be fragmented, biased, or poorly matched to the target modality.

Both goals are realized through the corpus-intrinsic design specified in the paper.

---

## Architecture at a Glance

PRISM is a corpus-intrinsic initialization pipeline for LDA with two stages:

1) **Corpus-derived embeddings**
- Compute a **PPMI** matrix from document-level co-occurrence (each document is a context window).
- Build a **word similarity graph** using cosine similarity between PPMI rows.
- Apply **diffusion maps** (anisotropic normalization, $t=1$) to obtain low-dimensional **word embeddings**.

2) **Prior estimation + LDA initialization**
- Fit a **GMM** over embeddings to obtain soft topic memberships $p(z\mid w)$, then convert to $p(w\mid z)$ via Bayes’ rule using unigram frequencies.
- Estimate a vocabulary-level **Dirichlet prior** $\hat{\beta}$ (method-of-moments).
- Initialize **MALLET LDA** with vector-valued $\hat{\beta}$, keeping the LDA generative process and inference unchanged.

---

### Overall Framework
<p align="center">
  <img src="figures/PRISM_Framework.png" width="90%" alt="PRISM full framework"/>
</p>

## Installation  

```bash
# 1) Clone repo

git clone https://github.com/tal-ishon/PRISM.git
cd PRISM

# 2) Run the setup script

chmod +x setup_mallet.sh
./setup_mallet.sh
```

PRISM requires a patched build of **MALLET 2.0.8** to support a **vector-valued $\hat \beta$** prior  (the setup script **downloads, patches, and builds** MALLET automatically)..

```bash
# 3) Set venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---

## Datasets
Datasets are expected under `data/` as `<DATASET>.json` files.

### Default layout
```
PRISM/
└── data/
      ├── 20NewsGroup.json
      ├── BBC.json
      ├── M10.json
      ├── DBLP.json
      └── TrumpTweets.json
```
Each json file is a JSON array of documents and each document is represented as a list of pre-tokenized word strings (e.g., `data/BBC.json`).

### References and Links
Below are the sources for the datasets used in our evaluation:

- ***20NewsGroup*** - https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/20NewsGroup (octis)
- ***BBC News*** - https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/BBC_News (octis)
- ***M10*** - https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/M10 (octis)
- ***DBLP*** - https://github.com/MIND-Lab/OCTIS/tree/master/preprocessed_datasets/DBLP (octis)
- ***Trump's Tweets*** - https://www.kaggle.com/datasets/codebreaker619/donald-trump-tweets-dataset (kaggle)


These links provide the original sources for the datasets referenced in the PRISM paper.

---

## Running the full pipeline (end-to-end)

To run PRISM from scratch, execute the pipeline in this order.

### 1) Create corpus-intrinsic word embeddings

This step computes the PPMI matrix, builds the word-similarity graph, and generates the prior inputs used later in the pipeline.

Example:
```bash
python create_word_embeddings.py \
  --n_comp 5 \
  --dataset BBC \
  --metrics prism \
  [OPTIONS]
```
**Required arguments:**
- `--n_comp`: Number of components/topics used when creating the prior
- `--dataset`: Dataset name, used to locate the saved prior files
- `--metric`: Metric name used in the saved prior filename, such as `prism` or `glove`
- `-h, --help`: Show the help message

### 2) Estimate the Dirichlet prior $\hat \beta$ (method of moments)

Example:

```bash
python methods_of_moments.py \
  --n_comp 5 \
  --dataset BBC \
  --metric prism
```

**Required arguments:**
- `--n_comp`: Number of components/topics used when creating the prior
- `--dataset`: Dataset name, used to locate the saved prior files
- `--metric`: Metric name used in the saved prior filename, such as `prism` or `glove`

For more information:
- `-h, --help`: Show the help message

### 3) Run LDA initialized with PRISM’s informative prior

Example:
```bash
python main.py \
  --n_comp 20 \
  --dataset 20NewsGroup \
  --metric prism \
  --num_iterations 1000 \
  --run_number FINAL \
  --seed 0
```
---

## 📚 BibTeX
```bibtex
@article{ishon2026prism,
  title={PRISM: PRIor from corpus Statistics for topicModeling},
  author={Ishon, Tal and Shaham, Uri},
  journal={arXiv preprint arXiv:2601.18900},
  year={2026}
}
```

---

<p align="center">Made for reliable topic discovery in data-scarce, under-characterized domains.</p>