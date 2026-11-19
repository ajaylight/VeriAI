# FEVER data files

Keep these files in this folder (do not commit them):
- `train.jsonl`
- `paper_dev.jsonl`
- `wiki-pages.zip` (required for BM25 evidence retrieval)

Official archive: https://fever.ai/download/fever/wiki-pages.zip
Official dataset page: https://fever.ai/dataset/fever.html

The compressed Wikipedia archive is about 1.7 GB and its JSONL contents are about 7.25 GB. VeriAI streams the JSONL files from the ZIP, so you do not need to create a second extracted copy. The SQLite full-text index can also require several GB; check available disk space before starting.

Each Wikipedia page record includes an `id` and a `lines` field. Each non-empty line in `lines` is formatted as `sentence_id<TAB>sentence_text`. VeriAI indexes each sentence separately and preserves the page ID and sentence ID for traceability.
