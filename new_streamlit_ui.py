import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, models
import torch

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
    index = faiss.read_index("data/bible_greek.index")
    metadata = pd.read_pickle("data/bible_metadata.pkl")
    return index, metadata

@st.cache_data
def load_translation_data():
    translation_df = pd.read_csv("data/og_lang_transformed.csv")
    return translation_df

@st.cache_data
def load_kjv_data():
    kjv_df = pd.read_csv("data/kjv.csv")
    return kjv_df

@st.cache_data
def load_verse_data():
    nt_split_column_df = pd.read_csv("data/nt_split_column_df.csv")
    nt_split_column_df['chapter'] = nt_split_column_df['chapter'].astype(int)
    nt_split_column_df['verse'] = nt_split_column_df['verse'].astype(int)
    return nt_split_column_df

@st.cache_resource
def load_faiss_index():
    """Load the pre-built FAISS index and its metadata."""
    index = faiss.read_index("data/bible_greek.index")
    metadata = pd.read_pickle("data/bible_metadata.pkl")
    return index, metadata

# Helper functions

model = load_model()
faiss_index, verse_metadata = load_faiss_index()

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

nt_split_column_df = load_verse_data()
translation_data = load_translation_data()
kjv_data = load_kjv_data()
books = sorted(translation_data['book'].dropna().unique())

# --sidebar--

with st.sidebar:
    st.title("📜 Navigation")
    book = st.selectbox("Select Book", books)
    
    # Filter chapters based on booK
    book_mask = translation_data['book'] == book
    chapter_tokens = translation_data.loc[book_mask, 'chapter'].dropna()
    chapter_list = sorted(chapter_tokens.unique().astype(int), key=int)
    chapter = st.selectbox("Select Chapter", chapter_list)
    
    # Filter verses based on chapter
    chapter_mask = book_mask & (translation_data['chapter'] == chapter)
    verse_tokens = translation_data.loc[chapter_mask, 'verse'].dropna()
    verse_list = sorted(verse_tokens[verse_tokens != ""].unique().astype(int), key=token_sort_key)
    verse = st.selectbox("Select Verse", verse_list)

# Get current verse data
verse_mask = (
    (translation_data['book'] == str(book))
    & (translation_data['chapter'] == chapter)
    & (translation_data['verse'] == verse)
)

verse_text_mask = (
    (nt_split_column_df['book'] == str(book).lower())
    & (nt_split_column_df['chapter'] == chapter)
    & (nt_split_column_df['verse'] == verse)
)

verse_kjv_mask = (
    (kjv_data['book_name'].str.lower() == str(book).lower())
    & (kjv_data['chapter_number'] == chapter)
    & (kjv_data['verse_number'] == verse)
)

current_verse_df = translation_data[verse_mask].sort_values('reference')
current_verse_text_df = nt_split_column_df[verse_text_mask]

# --main--

st.title("Verse Analysis")

st.write("## Verse Text")
# verse text
st.write(f"### Verse in Greek: \n{current_verse_text_df['text'].iloc[0]}")
st.write(f"### Verse in English (KJV): \n{kjv_data.loc[
    (kjv_data['book_name'].str.lower() == str(book).lower()) &
    (kjv_data['chapter_number'] == chapter) &
    (kjv_data['verse_number'] == verse),
    'verse_text'
].iloc[0]}")

# verse level analysis
st.write("## Verse-Level Analysis")
if 'button_checked' not in st.session_state:
    st.session_state.button_checked = False
def toggle_button():
    st.session_state.button_checked = not st.session_state.button_checked
st.button(
    label="Close Verse Analysis" if st.session_state.button_checked else "Open Verse Analysis",
    on_click=toggle_button,
    width='stretch'
)

if st.session_state.button_checked:
    st.write("### Most Similar Verses (Semantic Search)")
    if current_verse_text_df.empty:
        st.warning("No Greek verse text found for this selection.")
    else:
        verse_query = str(current_verse_text_df['text'].iloc[0]).strip()

        with st.spinner("Finding semantically similar verses..."):
            similar_df = semantic_search(verse_query, top_k=10)

        # Parse verse_ref into components and add KJV text
        parts = similar_df['verse_ref'].str.rsplit(' ', n=1)
        similar_df['book'] = parts.str[0]
        similar_df['ch_vs'] = parts.str[1]
        similar_df['chapter'] = similar_df['ch_vs'].str.split(':').str[0].astype(int)
        similar_df['verse'] = similar_df['ch_vs'].str.split(':').str[1].astype(int)
        similar_df['reference'] = similar_df['book'].str.title() + ' ' + similar_df['ch_vs']

        kjv_lookup = kjv_data[['book_name', 'chapter_number', 'verse_number', 'verse_text']].copy()
        kjv_lookup['book_lower'] = kjv_lookup['book_name'].str.lower()
        similar_df = similar_df.merge(
            kjv_lookup,
            left_on=['book', 'chapter', 'verse'],
            right_on=['book_lower', 'chapter_number', 'verse_number'],
            how='left',
        )

        display_cols = ["reference", "text", "verse_text", "similarity"]
        display_df = similar_df[display_cols].rename(columns={"text": "Greek", "verse_text": "KJV", "similarity": "Similarity"})
        st.dataframe(display_df, use_container_width=True)


st.write("## Word-Level Analysis")
for i, (_, row) in enumerate(current_verse_df.iterrows()):
    # Create columns for each row's data
    # Adjust the ratios [1, 2, 2] to fit your specific data widths
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        if st.button(row['text'], key=f"btn_{i}"):
            st.session_state.selected_greek = row['text']
            st.session_state.selected_english = row['translation']
            
    with col2:
        st.write(row['translation'])
        
    with col3:
        # Replace 'morphology' with your actual column name
        st.write(row['transliteration']) 
    
    st.divider() # Optional: adds a thin line between rows
