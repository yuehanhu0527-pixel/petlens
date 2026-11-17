
import math
import re
from collections import Counter, defaultdict

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "to", "of", "for", "with",
    "and", "or", "in", "on", "at", "by",
    "how", "why", "what", "when", "where",
    "should", "could", "would", "do", "does",
    "did", "i", "my", "me", "you", "your",
    "can", 
}

QUERY_EXPANSIONS = {
    "feed": {"feeding", "diet", "meal", "meals"},
    "feeding": {"feed", "meal", "meals"},
    "aggression": {"aggressive", "fighting"},
    "pee": {"urination", "urine", "urinating"},
    "litter": {"litterbox"},
}

def tokenize(text):
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.N = len(chunks)

        if self.N == 0:
            raise ValueError("The search corpus is empty.")

        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        self.document_frequency = Counter()

        total_length = 0

        for doc_id, chunk in enumerate(chunks):

            # Index the passage and its title
            text = (
                chunk["title"] + " " +
                chunk["title"] + " " +
                chunk["text"]
            )

            tokens = tokenize(text)
            term_counts = Counter(tokens)

            doc_length = len(tokens)

            self.doc_lengths.append(doc_length)
            total_length += doc_length

            # Inverted index: term -> [(doc_id, frequency)]
            for term, frequency in term_counts.items():

                self.inverted_index[term].append(
                    (doc_id, frequency)
                )

                self.document_frequency[term] += 1

        self.avgdl = total_length / self.N

        # Precompute IDF for every term
        self.idf = {}

        for term, df in self.document_frequency.items():

            self.idf[term] = math.log(
                1 + (self.N - df + 0.5) / (df + 0.5)
            )

    def search(self, query, top_k=5, category=None):

        if top_k <= 0:
            return []

        query_tokens = {
            token for token in tokenize(query)
            if token not in STOPWORDS
        }

        expanded_tokens = set(query_tokens)

        for token in query_tokens:
            expanded_tokens.update(
                QUERY_EXPANSIONS.get(token, set())
            )

        query_tokens = expanded_tokens

        if not query_tokens:
            return []

        scores = defaultdict(float)

        for term in query_tokens:

            if term not in self.inverted_index:
                continue

            idf = self.idf[term]

            for doc_id, tf in self.inverted_index[term]:

                chunk = self.chunks[doc_id]

                if (
                    category is not None
                    and chunk["category"] != category
                ):
                    continue

                dl = self.doc_lengths[doc_id]

                denominator = (
                    tf
                    + self.k1
                    * (
                        1 - self.b
                        + self.b * dl / self.avgdl
                    )
                )

                score = idf * (
                    tf * (self.k1 + 1)
                ) / denominator

                scores[doc_id] += score

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )

        results = []
        seen_documents = set()

        for doc_id, score in ranked:

            chunk = self.chunks[doc_id]
            document_id = chunk["document_id"]

            if document_id in seen_documents:
                continue

            seen_documents.add(document_id)

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": document_id,
                "title": chunk["title"],
                "category": chunk["category"],
                "authors": chunk["authors"],
                "url": chunk["url"],
                "text": chunk["text"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

