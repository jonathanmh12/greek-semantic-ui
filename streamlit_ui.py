import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, models

# --- CONFIG & MODEL LOADING ---
st.set_page_config(page_title="Greek Verse Explorer", layout="wide")

@st.cache_resource
def load_model():
    """Load the same model used to build the FAISS index."""
    word_embedding_model = models.Transformer("models/ancient-greek-biblical-sbert")
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode="mean",
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

@st.cache_resource
def load_faiss_index():
    """Load the pre-built FAISS index and its metadata."""
    index = faiss.read_index("bible_greek.index")
    metadata = pd.read_pickle("bible_metadata.pkl")
    return index, metadata

@st.cache_data
def load_data():
    dataset_url = "https://huggingface.co/datasets/hmcgovern/original-language-bibles-greek/resolve/main/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(dataset_url)
    return df

model = load_model()
faiss_index, verse_metadata = load_faiss_index()
df = load_data()
reference_data = df['reference'].astype(str).str.extract(
    r'^(?P<book>[^.]+)\.(?P<chapter>\d+)\.(?P<verse>.+)\.(?P<word>\d+)$'
)

# --- HELPER FUNCTIONS ---
def semantic_search(query: str, top_k: int = 10):
    """Search the FAISS index for verses most similar to `query`."""
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, top_k)
    results = verse_metadata.iloc[indices[0]].copy()
    results["similarity"] = distances[0]
    return results


def token_sort_key(token):
    token = str(token)
    numeric = ''.join(ch for ch in token if ch.isdigit())
    if numeric:
        return int(numeric), token
    return float('inf'), token

# --- DATA PARSING ---
# We simplify your build_verse_lookup into a clean nested structure
books = sorted(reference_data['book'].dropna().unique())

# --- SIDEBAR: SELECTION ---
with st.sidebar:
    st.title("📜 Navigation")
    book = st.selectbox("Select Book", books)
    
    # Filter chapters based on book
    book_mask = reference_data['book'] == book
    chapter_tokens = reference_data.loc[book_mask, 'chapter'].dropna()
    chapter_list = sorted(chapter_tokens.unique(), key=int)
    chapter = st.selectbox("Select Chapter", chapter_list)
    
    # Filter verses based on chapter
    chapter_mask = book_mask & (reference_data['chapter'] == str(chapter))
    verse_tokens = reference_data.loc[chapter_mask, 'verse'].dropna()
    verse_list = sorted(verse_tokens[verse_tokens != ""].unique(), key=token_sort_key)
    verse = st.selectbox("Select Verse", verse_list)

# Get current verse data
verse_mask = (
    (reference_data['book'] == str(book))
    & (reference_data['chapter'] == str(chapter))
    & (reference_data['verse'] == str(verse))
)
current_verse_df = df[verse_mask].sort_values('reference')

# --- MAIN UI ---
st.title(f"Greek Verse Explorer: {book} {chapter}:{verse}")

# Row 1: The Verse (Greek Words)
st.write("### Verse Text")
cols = st.columns(len(current_verse_df))

selected_word = None

for i, (_, row) in enumerate(current_verse_df.iterrows()):
    with cols[i]:
        # We use buttons to mimic the "clickable" word feature from your Pygame code
        if st.button(row['text'], key=f"word_{i}", help=row['translation']):
            st.session_state.selected_greek = row['text']
            st.session_state.selected_english = row['translation']

# Row 2: Analysis
st.markdown("---")

if 'selected_greek' in st.session_state:
    greek = st.session_state.selected_greek
    english = st.session_state.selected_english
    
    col_info, col_sim = st.columns([1, 2])
    
    with col_info:
        st.info(f"**Selected Word:** {greek}")
        st.write(f"**English Gloss:** {english}")
        
    with col_sim:
        st.write("### Similar Verses (FAISS)")
        with st.spinner("Searching vector database..."):
            results = semantic_search(greek, top_k=10)
            sim_df = results[["verse_ref", "text", "similarity"]].copy()
            sim_df.columns = ["Reference", "Verse Text", "Similarity"]
            sim_df = sim_df.reset_index(drop=True)
            st.dataframe(sim_df, use_container_width=True)
else:
    st.write("Click a Greek word above to view analysis and semantic tidbits.")