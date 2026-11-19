# Data preparation scripts

Run scripts from the `backend` directory so the `app` package is importable.

Build a small development index:

```cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip --max-pages 20000
```

Build the full index (long-running and disk-intensive):

```cmd
python -m app.scripts.build_fever_index --archive ..\data\fever\wiki-pages.zip
```

Use `--overwrite` only when you intend to rebuild an existing index. A successful build writes `data/index/fever_wiki_fts5.sqlite3`.
