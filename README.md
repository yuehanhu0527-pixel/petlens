# PetLens

**Evidence-Grounded Cat-Care Literature Search**  
CS 410 — Information Retrieval | Fall 2026

## Overview

PetLens is a small, research-oriented information retrieval system for finding scientific literature relevant to common cat-care questions. It searches a curated collection of feline-related articles and returns ranked text passages together with article titles, categories, similarity scores, and source links.

PetLens is a **literature search tool**, not a veterinary diagnosis system or a generative question-answering assistant. Retrieved passages may be related to a topic without directly answering the question. Users should consult a veterinarian for animal-health concerns.

## Features

- Interactive command-line search using a semantic retriever.
- Optional filtering by topic: `behavior`, `feeding`, `grooming`, `litter_box`, or `general_care`.
- Article-level result diversity: the default semantic search returns at most one passage per source article.
- Multiple retrieval approaches implemented for comparison: TF-IDF, BM25, semantic retrieval, and hybrid retrieval; an experimental cross-encoder reranker is also included.
- Evaluation scripts and saved baseline results for Precision@5, Mean Reciprocal Rank (MRR), and nDCG@5.
- Automated tests for the BM25, semantic, and hybrid retrieval components.

## Dataset

The curated corpus contains **74 feline-related research articles**, split into **1,637 passages**. Passage text and metadata are stored in `data/processed/chunks.json`; precomputed dense representations are stored in `data/processed/embeddings.npy`.

Articles were collected from Europe PMC / PubMed Central and processed through a collection, review, cleaning, and chunking pipeline. The collection process includes language, completeness, and open-license checks. Each search result includes a link to its original publication.

**Data and license note:** Individual articles retain their own attribution and license terms. Before redistributing full article text in a public repository, verify the applicable license and attribution requirements for every included article. Source links alone are not a substitute for required attribution. The repository's code license, if any, does not automatically apply to third-party articles.

## Requirements

- Python and `pip` in a virtual environment.
- Dependencies listed in `requirements.txt`.
- Internet access on the first run if the sentence-transformer model is not yet cached locally.

The project was developed and tested in a local Python 3.15 environment. Exact compatibility with other Python versions and operating systems has not been independently verified.

## Installation

Clone the project repository and open its root directory:

```bash
git clone https://github.com/yuehanhu0527-pixel/petlens.git
cd petlens
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The installed packages and model-loading behavior have primarily been checked on macOS.

**Before running the search**, make sure these files are available at the paths below:

```text
data/processed/chunks.json
data/processed/embeddings.npy
```

If the public repository does not include them for licensing or distribution reasons, it must provide an alternative documented way to obtain the processed data before the software can be run. The default search cannot run without both files.

## Run PetLens

From the repository root:

```bash
python -m app.main
```

This launches the interactive semantic-search interface. You can also launch that interface directly:

```bash
python -m scripts.search_semantic
```

Example interaction:

```text
Search: Why does my cat pee outside the litter box?
Categories: all, behavior, feeding, grooming, litter_box, general_care
Category [all]:
```

Press **Enter** to search all categories, or enter a category such as `litter_box`. Each result displays a title, category, cosine similarity score, source URL, and excerpt from the retrieved passage. Enter `quit` or `exit` at the search prompt to leave the program.

Additional retrieval interfaces are available for experiments:

```bash
python -m scripts.search_tfidf
python -m scripts.search_hybrid
python -m scripts.search_reranked
```

These experimental commands have not all been retested as part of the final-delivery smoke test. The primary demonstrated interface is the semantic search above.

## Implementation

The main implementation is organized as follows:

```text
app/
  main.py                  # CLI entry point
  retrieval/
    bm25.py                # BM25 lexical retrieval
    tfidf.py               # TF-IDF retrieval
    semantic.py            # Dense vector retrieval
    hybrid.py              # Combined retrieval experiment
    reranker.py            # Experimental cross-encoder reranking
scripts/
  collect_data.py          # Article collection
  curate_data.py           # Corpus review/curation
  clean_data.py            # Text preprocessing
  chunk_data.py            # Passage creation
  build_embeddings.py      # Dense vector generation
  search_semantic.py       # Interactive semantic search
  compare_retrievers.py    # Retrieval comparison
  evaluate_retrieval.py    # Evaluation pipeline
  ...
tests/                    # Automated retrieval tests
data/processed/            # Indexed passages and embeddings
data/evaluation/           # Judgments and experimental results
```

The semantic retriever uses the `sentence-transformers/all-MiniLM-L6-v2` model. It encodes a user's query, computes similarity against the stored passage embeddings, sorts candidate passages by score, and returns up to five results while preventing repeated articles in the default interface. The cross-encoder reranker is experimental and is **not** the default search method.

## Evaluation

Four retrieval approaches were compared using **10 representative cat-care queries** and an initial set of manually reviewed relevance judgments. The recorded baseline results are:

| Retrieval method | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| TF-IDF | 0.3000 | 0.5833 | 0.3065 |
| BM25 | 0.4800 | 0.7167 | 0.4757 |
| **Semantic** | **0.5200** | **0.9333** | **0.5574** |
| Hybrid | 0.4600 | 0.6833 | 0.4148 |

In this initial comparison, semantic retrieval performed best on all three reported metrics. These numbers are **exploratory**, not a claim of generalizable accuracy: the test set is small, the relevance labels are provisional, and further manual review could change the values. These results also predate some later metadata corrections; the stored baseline has not been recomputed to represent those updates.

Selected supporting files are in `data/evaluation/`, including `evaluation_metrics.csv` and relevance-judgment CSVs.

### Error analysis and limitations

1. **Limited corpus coverage:** The curated set does not cover every possible feline health or behavior question. For example, questions about eating non-food objects (*pica*) produced weak results.
2. **Topic similarity is not answer usefulness:** A study can mention the same concepts as a query without giving direct practical guidance. For example, a question about feeding frequency may retrieve feeding schedules used in a cat café.
3. **Out-of-scope queries:** The default retriever still returns the highest-scoring available passages even when none directly answers the query; similarity scores are not calibrated relevance probabilities.
4. **Research-article structure:** Experimental methods, publisher statements, and other non-answer content can sometimes appear in retrieved passages.
5. **Small evaluation set:** Ten queries and provisional labels do not establish performance on unrestricted user questions.

These issues are documented rather than hidden. PetLens is designed to make relevant research discoverable, not to generate clinical recommendations or guarantee an answer for each question.

## Testing

Run the automated test suite from the repository root:

```bash
python -m pytest -q
```

In the most recent local run, **24 tests passed**. The interactive semantic-search smoke test also successfully loaded all 1,637 passages and displayed ranked results with source links. These checks establish basic functionality, not clinical reliability or high retrieval precision.

## Reproducibility and project status

The current deliverable is the working command-line retrieval application, its implementation scripts, stored data artifacts, and experimental analysis. The code and documentation are intended to be evaluated based on their reproducibility, transparent limitations, and functioning search workflow.

The article collection and embedding-generation scripts are included for implementation transparency, but rebuilding the entire dataset may require external services, network access, and license review. For a straightforward demonstration, use the included processed passage and embedding files when distribution rights permit.

## Acknowledgments

PetLens uses research articles indexed by Europe PMC / PubMed Central and relies on open-source Python libraries for text processing and retrieval. Scientific findings belong to their original authors and publishers; consult each linked article for its complete citation and license details.
