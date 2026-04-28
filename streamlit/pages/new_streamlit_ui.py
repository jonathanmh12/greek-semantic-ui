import streamlit as st
import pandas as pd
import numpy as np
import faiss
import load_data_func

# Helper functions

model = load_data_func.load_model()
faiss_index, verse_metadata = load_data_func.load_faiss_index()
concepts_df, concept_embeddings = load_data_func.load_concept_bank()
topics_df, topic_embeddings = load_data_func.load_bertopic_concepts()

def concept_search(verse_text: str, top_k: int = 15):
    """Return the top_k curated concepts most similar to a Greek verse."""
    vec = model.encode([verse_text], convert_to_numpy=True)  # shape (1, 768)
    faiss.normalize_L2(vec)
    scores = concept_embeddings @ vec[0]  # (n_concepts,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    result = concepts_df.iloc[top_idx].copy()
    result["similarity"] = scores[top_idx]
    return result.reset_index(drop=True)

def topic_search(verse_text: str, top_k: int = 15):
    """Return the top_k BERTopic-discovered NT themes most similar to a Greek verse."""
    vec = model.encode([verse_text], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    scores = topic_embeddings @ vec[0]  # (n_topics,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    result = topics_df.iloc[top_idx].copy()
    result["similarity"] = scores[top_idx]
    return result.reset_index(drop=True)

def semantic_search(query: str, top_k: int = 100):
    """Search the FAISS index for verses most similar to `query`."""
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, top_k)
    results = verse_metadata.iloc[indices[0]].copy()
    results["similarity"] = distances[0]
    return results


nt_split_column_df = load_data_func.load_verse_data()
translation_data = load_data_func.load_translation_data()
kjv_data = load_data_func.load_kjv_data()
books = sorted(translation_data['book'].dropna().unique())

# --sidebar--

with st.sidebar:
    st.title("📜 Navigation")
    book = st.selectbox("Select Book", books)
    
    # Filter chapters based on book
    book_mask = translation_data['book'] == book
    chapter_tokens = translation_data.loc[book_mask, 'chapter'].dropna()
    chapter_list = sorted(chapter_tokens.unique().astype(int), key=int)
    
    st.subheader("Select Verse Range (up to 5 consecutive verses)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Starting Verse**")
        start_chapter = st.selectbox("Start Chapter", chapter_list, key="start_ch")
    
    with col2:
        st.write("**Ending Verse**")
        end_chapter = st.selectbox("End Chapter", chapter_list, key="end_ch", 
                                   index=min(chapter_list.index(start_chapter), len(chapter_list)-1))
    
    # Get verses for starting chapter
    start_chapter_mask = book_mask & (translation_data['chapter'] == start_chapter)
    start_verse_tokens = translation_data.loc[start_chapter_mask, 'verse'].dropna()
    start_verse_list = sorted(start_verse_tokens[start_verse_tokens != ""].unique().astype(int), key=load_data_func.token_sort_key)
    
    # Get verses for ending chapter
    end_chapter_mask = book_mask & (translation_data['chapter'] == end_chapter)
    end_verse_tokens = translation_data.loc[end_chapter_mask, 'verse'].dropna()
    end_verse_list = sorted(end_verse_tokens[end_verse_tokens != ""].unique().astype(int), key=load_data_func.token_sort_key)
    
    col1, col2 = st.columns(2)
    with col1:
        start_verse = st.selectbox("Start Verse", start_verse_list, key="start_vs")
    
    with col2:
        end_verse = st.selectbox("End Verse", end_verse_list, key="end_vs")
    
    # Generate the verse range
    selected_verses = load_data_func.get_verse_range(book, start_chapter, start_verse, end_chapter, end_verse, translation_data)
    
    if not selected_verses:
        st.error("Invalid verse range selection")
    else:
        st.success(f"✓ Successfully selected {len(selected_verses)} consecutive verse(s)")

# Get current verse data for all selected verses
current_verse_dfs = []
current_verse_text_dfs = []
current_verse_kjv_dfs = []

for chapter, verse in selected_verses:
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
    
    current_verse_dfs.append(translation_data[verse_mask].sort_values('reference'))
    current_verse_text_dfs.append(nt_split_column_df[verse_text_mask])
    current_verse_kjv_dfs.append(kjv_data[verse_kjv_mask])

# Consolidate all verses for analysis
consolidated_greek = " ".join([
    str(current_verse_text_dfs[idx]['text'].iloc[0]).strip() 
    for idx in range(len(selected_verses)) 
    if not current_verse_text_dfs[idx].empty
])

verse_label = f"{book} {selected_verses[0][0]}:{selected_verses[0][1]}"
if len(selected_verses) > 1:
    verse_label += f" - {book} {selected_verses[-1][0]}:{selected_verses[-1][1]}"

# --main--

st.title("Verse Analysis")

st.write("## Verse Text")

st.write(f"### 📖 {verse_label}")

# Build appended Greek and KJV texts across all selected verses
greek_parts = []
kjv_parts = []
for idx, (chapter, verse) in enumerate(selected_verses):
    verse_text_df = current_verse_text_dfs[idx]
    verse_kjv_df = current_verse_kjv_dfs[idx]
    ref = f"{chapter}:{verse}"
    if not verse_text_df.empty:
        greek_parts.append(f"[{ref}] {verse_text_df['text'].iloc[0].strip()}")
    if not verse_kjv_df.empty:
        kjv_parts.append(f"[{ref}] {verse_kjv_df['verse_text'].iloc[0].strip()}")

if greek_parts:
    st.write(f"**Greek:** {' '.join(greek_parts)}")
else:
    st.warning("No Greek verse text found.")

if kjv_parts:
    st.write(f"**KJV:** {' '.join(kjv_parts)}")
else:
    st.warning("No KJV verse text found.")

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
    st.write("### Most Similar NT Verses (Semantic Search)")
    
    if not consolidated_greek:
        st.warning("No Greek verse text found for selected verses.")
    else:
        with st.spinner("Finding semantically similar NT verses..."):
            similar_df = semantic_search(consolidated_greek, top_k=100)

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
    
    # concept mapping
    st.write("### English Concept Mapping")
    
    if not consolidated_greek:
        st.warning("No Greek verse text found for selected verses.")
    else:
        tab_curated, tab_discovered = st.tabs(["Curated Bank", "Discovered Topics"])

        with tab_curated:
            with st.spinner("Finding matching concepts..."):
                top_concepts = concept_search(consolidated_greek, top_k=15)
            top_concepts["Similarity"] = top_concepts["similarity"].map("{:.3f}".format)
            st.dataframe(
                top_concepts[["concept", "category", "Similarity"]].rename(
                    columns={"concept": "Concept", "category": "Category"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with tab_discovered:
            with st.spinner("Finding matching BERTopic themes..."):
                top_topics = topic_search(consolidated_greek, top_k=15)
            top_topics["Similarity"] = top_topics["similarity"].map("{:.3f}".format)
            st.dataframe(
                top_topics[["top_words", "count", "Similarity"]].rename(
                    columns={"top_words": "Theme Keywords", "count": "Verses in Cluster"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.info("LDA topic modeling analysis has moved to the Topic Modeling page.")