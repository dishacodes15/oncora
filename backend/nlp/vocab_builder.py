"""
vocab_builder.py
ONE-TIME / SCHEDULED job. Do NOT import this into the request path.

Pages through OpenFDA's /drug/ndc.json (the National Drug Code directory),
pulls every generic_name and brand_name it can find, normalizes them, and
writes the deduped set to a JSON file (drug_vocab.json). This file is what
parser.py loads at startup to build its PhraseMatcher.

IMPORTANT — pagination is partitioned by first character of generic_name.
OpenFDA hard-caps `skip` at 25000 regardless of total result count, and
the NDC directory has far more than 25000 entries total. A single
unfiltered query can never page past that ceiling. Splitting the query
into per-letter/digit slices (search=generic_name:a*, b*, ...) keeps each
slice's own result count low enough to fully page through.

Run manually:
    python -m backend.nlp.vocab_builder

Run on a schedule (weekly/monthly — the NDC directory doesn't change fast
enough to justify anything more frequent, and each full run is thousands
of paginated requests):
    0 3 * * 0  cd /app && python -m backend.nlp.vocab_builder   # cron, Sunday 3am
"""

import json
import logging
import os
import string
import time
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

from .openfda_client import search_ndc, OpenFDAError
from .vocab_filters import filter_vocab

load_dotenv()  # picks up OPENFDA_API_KEY from .env — this is a standalone
                # script, not imported by main.py, so it needs its own call

logger = logging.getLogger("oncora.nlp.vocab_builder")

PAGE_SIZE = 1000                      # OpenFDA's max limit per page
REQUEST_PAUSE_SECONDS = 0.3           # be a polite citizen even with a key
CHECKPOINT_EVERY_N_PAGES = 20         # write partial progress periodically
MAX_SKIP = 25000                      # OpenFDA's hard ceiling — going past
                                       # this in ANY single query 400s, no
                                       # matter how many total results exist

# Partition slices: a-z covers most generic_name values; a few products
# have purely numeric or symbol-leading names, so "0-9" and a catch-all
# pass with no filter (limited to the first MAX_SKIP records) cover the
# remainder without needing a slice per digit.
SLICE_PREFIXES = list(string.ascii_lowercase) + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def _normalize(name: str) -> str:
    """Lowercase + strip. We keep multi-word names intact (e.g. "acetylsalicylic
    acid") since PhraseMatcher handles multi-token phrases natively — no need
    to split them here."""
    return name.strip().lower()


def _harvest_page(results: list, vocab: Set[str]) -> None:
    for entry in results:
        generic = entry.get("generic_name")
        brand = entry.get("brand_name")
        if generic:
            vocab.add(_normalize(generic))
        if brand:
            vocab.add(_normalize(brand))


def _page_through_slice(search_expr: str, vocab: Set[str], label: str) -> None:
    """Pages through ONE slice (e.g. generic_name:a*) until exhausted or
    until MAX_SKIP is hit. Logs and moves on if a slice itself somehow has
    more than MAX_SKIP records — that would need further sub-partitioning,
    but is not expected for a single-letter generic_name prefix."""
    skip = 0
    while skip <= MAX_SKIP:
        try:
            data = search_ndc(limit=PAGE_SIZE, skip=skip, search=search_expr)
        except OpenFDAError as exc:
            logger.warning("Slice %r stopped early at skip=%d: %s", label, skip, exc)
            return

        results = data.get("results", [])
        if not results:
            logger.info("Slice %r exhausted at skip=%d", label, skip)
            return

        _harvest_page(results, vocab)
        total = data.get("meta", {}).get("results", {}).get("total")
        logger.info("Slice %r | skip=%d | +%d records | %d unique terms so far%s",
                    label, skip, len(results), len(vocab),
                    f" (slice total={total})" if total is not None else "")

        skip += PAGE_SIZE
        time.sleep(REQUEST_PAUSE_SECONDS)

        if total is not None and skip >= total:
            logger.info("Slice %r fully covered (%d/%d records)", label, skip, total)
            return

    logger.warning(
        "Slice %r hit MAX_SKIP=%d without exhausting — this slice has more "
        "records than one letter-prefix query can reach. Consider "
        "sub-partitioning (e.g. 'a*' -> 'aa*', 'ab*', ...) if coverage "
        "for this letter matters.", label, MAX_SKIP,
    )


def build_vocab(output_path: str, checkpoint_path: str = None) -> Set[str]:
    """
    Runs one paginated query per letter/digit prefix of generic_name,
    unions all the results into a single deduped vocabulary, and writes
    it to output_path as a sorted JSON list.

    checkpoint_path (optional): progress is saved after each completed
    slice, so a killed/interrupted run resumes from the next unfinished
    slice instead of starting over from scratch.
    """
    vocab: Set[str] = set()
    done_slices: Set[str] = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            state = json.load(f)
        vocab = set(state["vocab"])
        done_slices = set(state.get("done_slices", []))
        logger.info("Resuming from checkpoint: %d slices done, %d terms so far",
                    len(done_slices), len(vocab))

    for prefix in SLICE_PREFIXES:
        if prefix in done_slices:
            continue
        search_expr = f'generic_name:{prefix}*'
        try:
            _page_through_slice(search_expr, vocab, label=prefix)
        except Exception as exc:
            # Don't let one bad slice abort the whole run — log it and
            # move on. It'll just be retried on the next scheduled run
            # since it's not in done_slices.
            logger.error("Slice %r failed unexpectedly, skipping for this "
                         "run: %s", prefix, exc)
            continue
        done_slices.add(prefix)

        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({"vocab": sorted(vocab), "done_slices": sorted(done_slices)}, f)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    filtered_vocab = filter_vocab(vocab)
    dropped = len(vocab) - len(filtered_vocab)
    if dropped:
        logger.info("Filtered %d common-word false-positive terms before writing (%d remain)",
                    dropped, len(filtered_vocab))
    with open(output_path, "w") as f:
        json.dump(sorted(filtered_vocab), f, indent=2)
    logger.info("Wrote %d unique drug terms to %s", len(filtered_vocab), output_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return filtered_vocab


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = os.environ.get("DRUG_VOCAB_PATH", "data/drug_vocab.json")
    ckpt = os.environ.get("DRUG_VOCAB_CHECKPOINT", "data/.drug_vocab_checkpoint.json")
    build_vocab(output_path=out, checkpoint_path=ckpt)

