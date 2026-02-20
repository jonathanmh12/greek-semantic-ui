import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

# --- CONFIG & MODEL LOADING ---
st.set_page_config(page_title="Greek Verse Explorer", layout="wide")

@st.cache_resource
def load_model():
    model_dir = "models/GreekBERT_v3"
    transformer = models.Transformer(model_dir, tokenizer_args={"use_fast": True})
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[transformer, pooling])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

@st.cache_data
def load_data():
    dataset_url = "https://huggingface.co/datasets/hmcgovern/original-language-bibles-greek/resolve/main/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(dataset_url)
    return df

model = load_model()
df = load_data()
reference_data = df['reference'].astype(str).str.extract(
    r'^(?P<book>[^.]+)\.(?P<chapter>\d+)\.(?P<verse>.+)\.(?P<word>\d+)$'
)

# --- HELPER FUNCTIONS ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
        st.write("### Semantic Similarity")
        # Reuse your embedding logic
        with st.spinner("Finding similar words..."):
            target_emb = model.encode(greek)
            
            # For brevity/demo, we search within the current book's unique words
            book_words = df[df['reference'].str.startswith(f"{book}.")]['text'].unique()
            embeddings = model.encode(book_words)
            
            # Calculate similarities
            sims = [cosine_similarity(target_emb, e) for e in embeddings]
            results = sorted(zip(book_words, sims), key=lambda x: x[1], reverse=True)[1:11]
            
            # Display results in a table
            sim_df = pd.DataFrame(results, columns=["Greek Word", "Similarity Score"])
            sim_df = sim_df.astype({
                "Greek Word": object,
                "Similarity Score": "float64",
            })
            st.table(sim_df)
else:
    st.write("Click a Greek word above to view analysis and semantic tidbits.")