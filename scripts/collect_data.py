import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

OUTPUT = Path(__file__).resolve().parents[1] / "data/raw"

TARGET = 40
MAX_PAGES = 8

TOPICS = {
    "grooming": [
        "grooming",
        "overgrooming",
        "self-grooming",
        "hairball",
        "hairballs",
        "hair ball",
        "hair balls",
        "fur",
        "pruritus",
        "skin health",
        "flea allergy",
        "ear cleaning",
        "oral hygiene",
        "dental care",
        "tooth brushing",
        "periodontal disease"
    ],

    "litter_box": [
        "litter box",
        "litter boxes",
        "litter tray",
        "litter preference",
        "house-soiling",
        "house soiling",
        "urine marking",
        "urine spraying",
        "inappropriate elimination",
        "inappropriate urination",
        "periuria",
        "latrining",
        "toileting",
        "idiopathic cystitis"
    ]
}


SESSION = requests.Session()

SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
    )
)

SESSION.headers["User-Agent"] = (
    "PetLens/1.0 (educational information retrieval project)"
)


def get(path, **params):

    response = SESSION.get(
        f"{BASE}/{path}",
        params=params,
        timeout=45
    )

    response.raise_for_status()

    time.sleep(0.35)

    return response


def clean(value):
    return " ".join(value.split())


def node_text(node):
    if node is None:
        return ""

    return clean("".join(node.itertext()))


def read_article(record, category):

    pmcid = record["pmcid"]

    root = ET.fromstring(
        get(f"{pmcid}/fullTextXML").content
    )

    # Check language
    language = root.get(
        "{http://www.w3.org/XML/1998/namespace}lang",
        "en"
    )

    if language.lower().split("-")[0] != "en":
        return None

    # Extract article metadata
    meta = root.find("./front/article-meta")

    if meta is None:
        return None

    # Check copyright license
    license_node = meta.find("./permissions/license")

    if license_node is None:
        return None

    links = [
        value
        for node in license_node.iter()
        for key, value in node.attrib.items()
        if key.endswith("href")
    ]

    links += re.findall(
        r"https?://creativecommons\.org/[^\s<>\)]+",
        node_text(license_node)
    )

    allowed = next(
        (
            link.rstrip(".,;")
            for link in links
            if re.fullmatch(
                r"https?://creativecommons\.org/"
                r"(?:licenses/by/[0-9.]+"
                r"|publicdomain/zero/1\.0)/?",
                link.rstrip(".,;")
            )
        ),
        None
    )

    if not allowed:
        return None

    # Extract article body
    body = root.find("./body")

    if body is None:
        return None

    paragraphs = []

    def walk(node):

        excluded_tags = {
            "ref-list",
            "fig",
            "table-wrap",
            "supplementary-material",
            "ack",
            "boxed-text"
        }

        if node.tag in excluded_tags:
            return

        if node.tag == "p":

            value = node_text(node)

            if len(value) >= 80:
                paragraphs.append(value)

            return

        for child in node:
            walk(child)

    walk(body)

    text = "\n\n".join(paragraphs)

    # Filter incomplete articles
    if (
        len(text) < 1500
        or not record.get("authorString")
        or not record.get("pubYear")
    ):
        return None

    return {
        "document_id": pmcid,
        "title": record.get("title", ""),
        "source": (
            record.get("journalInfo", {})
            .get("journal", {})
            .get("title", "")
        ),
        "authors": record["authorString"],
        "year": record["pubYear"],
        "doi": record.get("doi", ""),
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "category": category,
        "license": allowed,
        "license_notice": node_text(license_node),
        "copyright_notice": node_text(
            meta.find("./permissions/copyright-statement")
        ),
        "abstract": record.get("abstractText", ""),
        "text": text
    }


EXCLUDED_TITLE_TERMS = [
    "pharmacokinetic",
    "phylogenetic",
    "genotypic",
    "molecular detection",
    "molecular characterization",
    "transcriptional regulation",
    "gene expression",
    "histopathological",
    "pathogenesis",
    "virus challenge",
    "metabolomics",
    "proteomics",
    "transcriptomic",
    "bioavailability",
    "case report",
    "case series",
    "pilot discovery",
    "alzheimer",
    "human aging"
]


def is_relevant(record, terms):

    title = (record.get("title") or "").lower()
    abstract = (record.get("abstractText") or "").lower()

    if any(
        term in title
        for term in EXCLUDED_TITLE_TERMS
    ):
        return False

    content = title + " " + abstract

    return any(
        term.lower() in content
        for term in terms
    )


def save_data(articles):

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = [
        {
            key: article.get(key, "")
            for key in (
                "document_id",
                "title",
                "source",
                "authors",
                "year",
                "category",
                "url",
                "license",
                "abstract"
            )
        }
        for article in articles
    ]

    files = {
        "articles_candidates.json": articles,
        "review_manifest.json": summary
    }

    for filename, content in files.items():

        destination = OUTPUT / filename
        temporary = OUTPUT / (filename + ".tmp")

        temporary.write_text(
            json.dumps(
                content,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        temporary.replace(destination)


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    existing_file = OUTPUT / "articles_candidates.json"

    # Load previously downloaded articles
    if existing_file.exists():

        articles = json.loads(
            existing_file.read_text(encoding="utf-8")
        )

    else:
        articles = []

    seen = {
        article["document_id"]
        for article in articles
    }

    initial_count = len(articles)

    print("Previously downloaded:", initial_count)

    for category, terms in TOPICS.items():

        count = sum(
            article["category"] == category
            for article in articles
        )

        print(f"\nSearching {category}...")
        print(f"Existing articles: {count}")

        if count >= TARGET:
            continue

        topic_query = " OR ".join(
            f'TITLE_ABS:"{term}"'
            for term in terms
        )

        query = (
            "(TITLE:cat OR TITLE:cats OR TITLE:feline "
            "OR TITLE:kitten OR TITLE:kittens) "
            f"AND ({topic_query}) "
            "AND OPEN_ACCESS:Y "
            "AND NOT SRC:PPR "
            'AND NOT PUB_TYPE:"retracted publication" '
            'AND NOT PUB_TYPE:"retraction of publication"'
        )

        cursor = "*"
        page = 0

        while count < TARGET and page < MAX_PAGES:

            page += 1

            print(f"  Searching page {page}...", flush=True)

            try:

                response = get(
                    "search",
                    query=query,
                    format="json",
                    resultType="core",
                    pageSize=100,
                    cursorMark=cursor
                )

                payload = response.json()

            except (
                requests.RequestException,
                ValueError
            ) as exc:

                print(f"Search failed: {exc}")
                break

            results = payload.get(
                "resultList", {}
            ).get("result", [])

            if page == 1:
                print("API matches:", payload.get("hitCount"))

            if not results:
                break

            for record in results:

                pmcid = record.get("pmcid")

                journal = (
                    record.get("journalInfo", {})
                    .get("journal", {})
                    .get("title", "")
                )

                if (
                    not pmcid
                    or pmcid in seen
                    or not journal
                ):
                    continue

                if not is_relevant(record, terms):
                    continue

                try:

                    article = read_article(
                        record,
                        category
                    )

                except (
                    requests.RequestException,
                    ET.ParseError
                ) as exc:

                    print(f"Skipped {pmcid}: {exc}")
                    continue

                if article is None:
                    continue

                articles.append(article)
                seen.add(pmcid)

                count += 1

                print(
                    f'  {count}/{TARGET}: {article["title"]}',
                    flush=True
                )

                save_data(articles)

                if count >= TARGET:
                    break

            next_cursor = payload.get("nextCursorMark")

            if not next_cursor or next_cursor == cursor:
                break

            cursor = next_cursor

        print(f"{category}: {count} candidates")

    save_data(articles)

    print("\n" + "=" * 50)
    print("Previously downloaded:", initial_count)
    print("New articles:", len(articles) - initial_count)
    print("Total candidates:", len(articles))
    print("Saved to:", OUTPUT)
    print("=" * 50)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

