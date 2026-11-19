# FEVER dataset files

Place the official labelled FEVER JSONL files here:
- `train.jsonl` — labelled training claims and evidence annotations.
- `paper_dev.jsonl` — labelled development claims and evidence annotations.

Downloads:
- https://fever.ai/download/fever/train.jsonl
- https://fever.ai/download/fever/paper_dev.jsonl

Each line is a JSON object. The loader preserves claim ID, text, label, and nested
evidence annotations. Evidence annotations reference Wikipedia page names and
sentence IDs; they are not the evidence text itself.

Do not commit dataset files. They are excluded by .gitignore. The large Wikipedia
corpus is deferred until retrieval development.
