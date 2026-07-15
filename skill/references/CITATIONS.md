# How the citation / hallucination guard works

You cannot force a model to never hallucinate. muck instead makes every claim **cheaply and
deterministically falsifiable**, and makes verification automatic and mandatory.

## Citation tokens
Every search hit and entity mention carries a token:

```
f2b3b7bde0f.0@40-59#6e80724b
└── doc_id ──┘ │   │   └ short hash of the exact source span
               └ char offsets into the document's text
```

The hash binds the exact span text. It is **tamper-evident**: a token only verifies if the
text at those offsets in the source hashes to that suffix.

## `muck cite <token>`
Re-derives the document text **from the original source file** — re-running the mapper for a
JSON record, or re-parsing the PDF — slices `[start:end]`, recomputes the hash, and returns
the span plus surrounding context. Fields reported:
- `token_valid` — the hash matches the re-derived span (token not fabricated/stale).
- `source_available` / `source_confirmed` — the original file was re-read and its span matches
  the indexed copy (catches index drift, not just model error).
- `valid` — both hold.

## `muck verify --token <t> --quote "<q>"`
Resolves the span, then checks the claimed quote actually occurs within it (whitespace- and
case-insensitive). Returns `quote_supported` and an overall `verified`. A fabricated quote
yields `quote_supported: false`; a tampered token yields `token_valid: false`. Either way,
`verified: false`.

## Findings ledger + `muck audit`
`muck finding add` stores `{claim, quote, citation_token}` and verifies on entry.
`muck audit` re-verifies **every** finding against source and reports green/red per claim —
the trace an editor reads instead of re-doing the work. Statuses: `verified`,
`unsupported` (quote not at the span), `invalid_token`.

## Text provenance — what "verified" means for scanned documents
For native text, `verify` proves *the document says this*. For a scanned document, the
document is **pixels** — so a citation carries `text_provenance` and states honestly what
verification covers:

- **native** — extracted from a real text layer. `verify` re-derives from the source file;
  `verified: true` means the document says it. (Unchanged.)
- **ocr** — a deterministic engine (tesseract) transcribed the page image. `verify` re-derives
  from the transcript **sidecar** (byte-stable, so hashes hold), and reports `verification_scope:
  transcript` + a `page_image`. OCR can misread a character but cannot invent a sentence, so
  `verified: true` is sound for the *transcript*; a `caveat` flags possible misreads.
- **vision** — an LLM transcribed the pixels, and an LLM *can* fabricate. `verify` returns
  `verified: false` and `needs_pixel_review: true` **even when the quote matches the transcript**
  — quote-in-transcript is necessary but not sufficient. The finding is `pending_review` until a
  human/agent confirms the `page_image` with `muck review <id> --confirm|--reject`. This is the
  one place muck refuses to let "the model agrees with the model" pass as verification.

`muck audit` buckets findings by provenance and surfaces a `pixel_review_queue`, so an editor
sees exactly which claims still rest on unreviewed pixels. Every non-native citation resolves to
the page image, so verification is always checkable against the original scan.

## Why JSON corpora are especially strong
A citation points to an exact field of an exact record (`file#/results/417`), and aggregate
findings are backed by reproducible SQL over typed values rather than fuzzy text — so totals
and counts are exact and re-runnable, not paraphrased.

## The discipline (enforced in SKILL.md)
Retrieve → quote verbatim → `muck verify` → `muck finding add` → `muck audit` before
presenting. No claim without a passing citation; say "unsupported" otherwise.
