# Greek Semantic UI

Interactive UI for exploring Greek semantic embeddings from the Bible.

## Overview

This application provides a Pygame-based interface for exploring semantic relationships in Biblical Greek text. Select any Greek word to see its closest semantic neighbors based on embedding similarity.

## Project Structure

```
├── test_ui.py         # Main Pygame application
├── models/            # Model files (symlink or copy from training repo)
└── pyproject.toml
```

## Setup

```bash
uv sync
```

## Usage

```bash
python test_ui.py
```

For corpus-wide word retrieval, place your training text at `greek_corpus.txt` in the project root. The app extracts unique Greek word tokens from that file and uses them as similarity candidates.

**How it works:**
1. Load a pre-trained Greek embedding model
2. Generate embeddings for your corpus
3. Click on Greek words to see semantically similar terms
4. Cosine similarity determines ranking

## Data

Currently loads verse data from John 1:1 for clickable UI text. Similarity candidates come from `greek_corpus.txt` (if present), plus visible Greek tokens in `verse_data`.

## Models

Place trained models in the `models/` directory. Currently configured to load from `models/Greek_v2_Expanded`.

## Next Steps

- Load from entire Greek corpus for broader similarity matching
- Add filtering by part of speech
- Implement search functionality
- Support batch similarity comparisons
