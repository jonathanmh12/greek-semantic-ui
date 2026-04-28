import faiss
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_model():
    """Load greek-nt-sbert_v2 (Transformer + CLS pooling + Dense + Normalize)."""
    model = SentenceTransformer("models/greek-nt-sbert_v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """Load the pre-built FAISS index and its metadata for greek-nt-sbert_v2."""
    index = faiss.read_index("data/greek-nt-sbert_v2/bible_greek.index")
    metadata = pd.read_pickle("data/greek-nt-sbert_v2/bible_metadata.pkl")
    return index, metadata

@st.cache_resource(show_spinner=False)
def load_concept_bank():
    """Load the curated concept bank and its pre-computed unit-norm embeddings."""
    concepts_df = pd.read_pickle("data/greek-nt-sbert_v2/concept_bank.pkl")
    embeddings = np.load("data/greek-nt-sbert_v2/concept_embeddings.npy")
    return concepts_df, embeddings

@st.cache_resource(show_spinner=False)
def load_bertopic_concepts():
    """Load BERTopic-discovered NT themes and their pre-computed embeddings."""
    topics_df = pd.read_pickle("data/greek-nt-sbert_v2/bertopic_terms.pkl")
    embeddings = np.load("data/greek-nt-sbert_v2/bertopic_topic_embeddings.npy")
    return topics_df, embeddings

@st.cache_data(show_spinner=False)
def load_translation_data():
    translation_df = pd.read_csv("data/og_lang_transformed.csv")
    return translation_df

@st.cache_data(show_spinner=False)
def load_kjv_data():
    kjv_df = pd.read_csv("data/kjv.csv")
    return kjv_df

@st.cache_data(show_spinner=False)
def load_verse_data():
    nt_split_column_df = pd.read_csv("data/nt_split_column_df.csv")
    nt_split_column_df['chapter'] = nt_split_column_df['chapter'].astype(int)
    nt_split_column_df['verse'] = nt_split_column_df['verse'].astype(int)
    return nt_split_column_df

def token_sort_key(token):
    token = str(token)
    numeric = ''.join(ch for ch in token if ch.isdigit())
    if numeric:
        return int(numeric), token
    return float('inf'), token

def get_verse_range(book_name, start_chapter, start_verse, end_chapter, end_verse, data):
    """Generate list of (chapter, verse) tuples from start to end, crossing chapters if needed."""
    verses = []
    current_chapter = start_chapter

    book_chapters = sorted(data[data['book'] == book_name]['chapter'].dropna().unique().astype(int))

    while len(verses) < 5:
        chapter_verses = sorted(
            data[(data['book'] == book_name) & (data['chapter'] == current_chapter)]['verse']
            .dropna()[lambda x: x != ""].unique().astype(int),
            key=int
        )

        if not chapter_verses:
            break

        if current_chapter == start_chapter and current_chapter == end_chapter:
            for v in chapter_verses:
                if start_verse <= v <= end_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
            break
        elif current_chapter == start_chapter:
            for v in chapter_verses:
                if v >= start_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
        elif current_chapter == end_chapter:
            for v in chapter_verses:
                if v <= end_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
            break
        else:
            for v in chapter_verses:
                if len(verses) < 5:
                    verses.append((current_chapter, v))

        chapter_idx = book_chapters.index(current_chapter)
        if chapter_idx + 1 < len(book_chapters):
            current_chapter = book_chapters[chapter_idx + 1]
        else:
            break

    return verses