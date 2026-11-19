# PetLens Retrieval Analysis

## Semantic Search Failure Case 1

**Query:** Why does my cat keep scratching its ears?

**Observed behavior:** Semantic Search returned passages about animal welfare, environmental enrichment, and feline urine spraying among its top results.

**Relevant document present in corpus:**
*Prevalence of Otitis Externa in a Population of Owned Cats in Northern Italy.*

**Corpus inspection:**
- `otitis`: 16 matching passages
- `ear infection`: 1 matching passage
- `ear scratching`: 1 matching passage
- `ear mites`: 0 matching passages
- `itchy ears`: 0 matching passages

**Interpretation:** The corpus contains potentially relevant information, but the semantic retriever did not surface it in the inspected top results.

**Possible contributing factors:** General-purpose embedding model, long passage truncation, and semantic mismatch between everyday language and veterinary terminology.
