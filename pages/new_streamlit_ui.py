import streamlit as st
import pandas as pd
import numpy as np
import faiss
import altair as alt
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

@st.cache_resource
def load_model():
    """Load greek-nt-sbert_v2 (Transformer + CLS pooling + Dense + Normalize)."""
    model = SentenceTransformer("models/greek-nt-sbert_v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

@st.cache_resource
def load_faiss_index():
    """Load the pre-built FAISS index and its metadata for greek-nt-sbert_v2."""
    index = faiss.read_index("data/greek-nt-sbert_v2/bible_greek.index")
    metadata = pd.read_pickle("data/greek-nt-sbert_v2/bible_metadata.pkl")
    return index, metadata

@st.cache_resource
def load_concept_bank():
    """Load the curated concept bank and its pre-computed unit-norm embeddings."""
    concepts_df = pd.read_pickle("data/greek-nt-sbert_v2/concept_bank.pkl")
    embeddings = np.load("data/greek-nt-sbert_v2/concept_embeddings.npy")
    return concepts_df, embeddings

@st.cache_resource
def load_bertopic_concepts():
    """Load BERTopic-discovered NT themes and their pre-computed embeddings."""
    topics_df = pd.read_pickle("data/greek-nt-sbert_v2/bertopic_terms.pkl")
    embeddings = np.load("data/greek-nt-sbert_v2/bertopic_topic_embeddings.npy")
    return topics_df, embeddings

@st.cache_data
def load_lda_artifacts():
    """Load precomputed LDA artifacts for fast topic visualizations at runtime."""
    lda_dir = Path("data/greek-nt-sbert_v2/lda")
    required_paths = {
        "topic_labels": lda_dir / "topic_labels.csv",
        "topic_summary": lda_dir / "topic_summary.csv",
        "topic_term_weights": lda_dir / "topic_term_weights.csv",
        "verse_topics": lda_dir / "verse_topics.csv",
    }
    optional_paths = {
        "subtopic_summary": lda_dir / "subtopic_summary.csv",
        "subtopic_term_weights": lda_dir / "subtopic_term_weights.csv",
    }

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        return None

    topic_labels_df = pd.read_csv(required_paths["topic_labels"])
    topic_summary_df = pd.read_csv(required_paths["topic_summary"])
    topic_term_weights_df = pd.read_csv(required_paths["topic_term_weights"])
    verse_topics_df = pd.read_csv(required_paths["verse_topics"])
    subtopic_summary_df = (
        pd.read_csv(optional_paths["subtopic_summary"])
        if optional_paths["subtopic_summary"].exists()
        else pd.DataFrame()
    )
    subtopic_term_weights_df = (
        pd.read_csv(optional_paths["subtopic_term_weights"])
        if optional_paths["subtopic_term_weights"].exists()
        else pd.DataFrame()
    )

    topic_labels_df["topic"] = topic_labels_df["topic"].astype(int)
    topic_summary_df["topic"] = topic_summary_df["topic"].astype(int)
    topic_term_weights_df["topic"] = topic_term_weights_df["topic"].astype(int)
    topic_term_weights_df["rank"] = topic_term_weights_df["rank"].astype(int)
    if "dominant_topic" in verse_topics_df.columns:
        verse_topics_df["dominant_topic"] = verse_topics_df["dominant_topic"].astype(int)
    if "subtopic" in verse_topics_df.columns:
        verse_topics_df["subtopic"] = verse_topics_df["subtopic"].astype(int)
    if "topic_path" in verse_topics_df.columns:
        verse_topics_df["topic_path"] = verse_topics_df["topic_path"].astype("string")
    verse_topics_df["verse_ref_key"] = verse_topics_df["verse_ref"].astype(str).str.strip().str.lower()

    if not subtopic_summary_df.empty:
        subtopic_summary_df["parent_topic"] = subtopic_summary_df["parent_topic"].astype(int)
        subtopic_summary_df["subtopic"] = subtopic_summary_df["subtopic"].astype(int)
        subtopic_summary_df["topic_path"] = subtopic_summary_df["topic_path"].astype("string")
    if not subtopic_term_weights_df.empty:
        subtopic_term_weights_df["parent_topic"] = subtopic_term_weights_df["parent_topic"].astype(int)
        subtopic_term_weights_df["subtopic"] = subtopic_term_weights_df["subtopic"].astype(int)
        subtopic_term_weights_df["rank"] = subtopic_term_weights_df["rank"].astype(int)
        subtopic_term_weights_df["topic_path"] = subtopic_term_weights_df["topic_path"].astype("string")

    return {
        "topic_labels": topic_labels_df,
        "topic_summary": topic_summary_df,
        "topic_term_weights": topic_term_weights_df,
        "verse_topics": verse_topics_df,
        "subtopic_summary": subtopic_summary_df,
        "subtopic_term_weights": subtopic_term_weights_df,
    }

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

# Helper functions

model = load_model()
faiss_index, verse_metadata = load_faiss_index()
concepts_df, concept_embeddings = load_concept_bank()
topics_df, topic_embeddings = load_bertopic_concepts()
lda_artifacts = load_lda_artifacts()

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


def token_sort_key(token):
    token = str(token)
    numeric = ''.join(ch for ch in token if ch.isdigit())
    if numeric:
        return int(numeric), token
    return float('inf'), token

def to_verse_ref_key(book_name, chapter, verse):
    normalized_book = re.sub(r"\s+", " ", str(book_name).strip().lower())
    return f"{normalized_book} {int(chapter)}:{int(verse)}"

def topic_label(topic_id, top_terms, max_terms=5):
    if not isinstance(top_terms, str) or not top_terms.strip():
        return f"Topic {topic_id}"
    terms = [term.strip() for term in top_terms.split(",") if term.strip()]
    return f"Topic {topic_id}: {', '.join(terms[:max_terms])}"


def topic_path_label(topic_path, top_terms, max_terms=4):
    path_text = str(topic_path)
    if not isinstance(top_terms, str) or not top_terms.strip():
        return f"Path {path_text}"
    terms = [term.strip() for term in top_terms.split(",") if term.strip()]
    return f"Path {path_text}: {', '.join(terms[:max_terms])}"

def get_verse_range(book_name, start_chapter, start_verse, end_chapter, end_verse, data):
    """Generate list of (chapter, verse) tuples from start to end, crossing chapters if needed."""
    verses = []
    current_chapter = start_chapter
    current_verse = start_verse
    
    # Get all chapters for the book
    book_chapters = sorted(data[data['book'] == book_name]['chapter'].dropna().unique().astype(int))
    
    while len(verses) < 5:
        # Get verses for current chapter
        chapter_verses = sorted(
            data[(data['book'] == book_name) & (data['chapter'] == current_chapter)]['verse']
            .dropna()[lambda x: x != ""].unique().astype(int),
            key=int
        )
        
        if not chapter_verses:
            break
            
        # Add verses from current chapter
        if current_chapter == start_chapter and current_chapter == end_chapter:
            # Same chapter, from start_verse to end_verse
            for v in chapter_verses:
                if start_verse <= v <= end_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
            break
        elif current_chapter == start_chapter:
            # First chapter, from start_verse to end of chapter
            for v in chapter_verses:
                if v >= start_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
        elif current_chapter == end_chapter:
            # Last chapter, from start of chapter to end_verse
            for v in chapter_verses:
                if v <= end_verse and len(verses) < 5:
                    verses.append((current_chapter, v))
            break
        else:
            # Middle chapter, add all verses
            for v in chapter_verses:
                if len(verses) < 5:
                    verses.append((current_chapter, v))
        
        # Move to next chapter
        chapter_idx = book_chapters.index(current_chapter)
        if chapter_idx + 1 < len(book_chapters):
            current_chapter = book_chapters[chapter_idx + 1]
        else:
            break
    
    return verses

nt_split_column_df = load_verse_data()
translation_data = load_translation_data()
kjv_data = load_kjv_data()
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
    start_verse_list = sorted(start_verse_tokens[start_verse_tokens != ""].unique().astype(int), key=token_sort_key)
    
    # Get verses for ending chapter
    end_chapter_mask = book_mask & (translation_data['chapter'] == end_chapter)
    end_verse_tokens = translation_data.loc[end_chapter_mask, 'verse'].dropna()
    end_verse_list = sorted(end_verse_tokens[end_verse_tokens != ""].unique().astype(int), key=token_sort_key)
    
    col1, col2 = st.columns(2)
    with col1:
        start_verse = st.selectbox("Start Verse", start_verse_list, key="start_vs")
    
    with col2:
        end_verse = st.selectbox("End Verse", end_verse_list, key="end_vs")
    
    # Generate the verse range
    selected_verses = get_verse_range(book, start_chapter, start_verse, end_chapter, end_verse, translation_data)
    
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

    st.write("### LDA Topic Insights (Precomputed)")
    if lda_artifacts is None:
        st.info(
            "Precomputed LDA artifacts are missing. Run `python build_lda_artifacts.py` once to generate them."
        )
    else:
        topic_summary_df = lda_artifacts["topic_summary"].copy()
        topic_term_weights_df = lda_artifacts["topic_term_weights"].copy()
        verse_topics_df = lda_artifacts["verse_topics"].copy()
        subtopic_summary_df = lda_artifacts.get("subtopic_summary", pd.DataFrame()).copy()
        subtopic_term_weights_df = lda_artifacts.get("subtopic_term_weights", pd.DataFrame()).copy()

        has_subtopic_assignments = {"subtopic", "subtopic_confidence", "topic_path"}.issubset(
            verse_topics_df.columns
        )
        has_subtopic_summary = not subtopic_summary_df.empty
        has_subtopic_term_weights = not subtopic_term_weights_df.empty

        topic_summary_df["topic_label"] = topic_summary_df.apply(
            lambda row: topic_label(row["topic"], row.get("top_terms", "")),
            axis=1,
        )

        col_overview, col_topic_terms = st.columns([1.5, 1])

        with col_overview:
            st.caption("Corpus topic prevalence")
            prevalence_chart = (
                alt.Chart(topic_summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(
                        "topic_label:N",
                        title="Topic",
                        axis=alt.Axis(labelAngle=-20, labelLimit=260),
                    ),
                    y=alt.Y("verse_count:Q", title="Verses"),
                    color=alt.Color(
                        "avg_confidence:Q",
                        title="Avg Confidence",
                        scale=alt.Scale(scheme="tealblues"),
                    ),
                    tooltip=[
                        alt.Tooltip("topic:Q", title="Topic"),
                        alt.Tooltip("verse_count:Q", title="Verse Count", format=","),
                        alt.Tooltip("verse_share:Q", title="Corpus Share", format=".2%"),
                        alt.Tooltip("avg_confidence:Q", title="Avg Confidence", format=".3f"),
                        alt.Tooltip("top_terms:N", title="Top Terms"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(prevalence_chart, use_container_width=True)

        with col_topic_terms:
            st.caption("Topic parts (top stems)")
            topic_options = topic_summary_df["topic"].tolist()
            selected_topic = st.selectbox(
                "Inspect Topic",
                topic_options,
                format_func=lambda topic_id: topic_label(
                    topic_id,
                    topic_summary_df.loc[topic_summary_df["topic"] == topic_id, "top_terms"].iloc[0],
                    max_terms=3,
                ),
                key="lda_topic_select",
            )

            selected_term_weights = (
                topic_term_weights_df[topic_term_weights_df["topic"] == selected_topic]
                .sort_values("rank")
                .head(12)
                .copy()
            )
            selected_term_weights["term_rank_label"] = (
                selected_term_weights["rank"].astype(str) + ". " + selected_term_weights["term"]
            )

            terms_chart = (
                alt.Chart(selected_term_weights)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("weight:Q", title="Weight"),
                    y=alt.Y(
                        "term_rank_label:N",
                        sort=alt.SortField("rank", order="ascending"),
                        title="Top Terms",
                    ),
                    color=alt.value("#2f7f5f"),
                    tooltip=[
                        alt.Tooltip("topic:Q", title="Topic"),
                        alt.Tooltip("rank:Q", title="Rank"),
                        alt.Tooltip("term:N", title="Term"),
                        alt.Tooltip("weight:Q", title="Weight", format=".3f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(terms_chart, use_container_width=True)

        st.caption("Dominant topic per selected verse")
        selected_ref_rows = [
            {
                "reference": f"{book} {chapter}:{verse}",
                "verse_ref_key": to_verse_ref_key(book, chapter, verse),
            }
            for chapter, verse in selected_verses
        ]
        selected_ref_df = pd.DataFrame(selected_ref_rows)

        selected_topics_df = selected_ref_df.merge(
            verse_topics_df,
            on="verse_ref_key",
            how="left",
        )
        selected_topics_df = selected_topics_df.merge(
            topic_summary_df[["topic", "top_terms"]],
            left_on="dominant_topic",
            right_on="topic",
            how="left",
        )

        if has_subtopic_assignments and has_subtopic_summary:
            subtopic_lookup_df = (
                subtopic_summary_df[["topic_path", "top_terms"]]
                .rename(columns={"top_terms": "subtopic_terms"})
                .drop_duplicates(subset=["topic_path"])
            )
            selected_topics_df["topic_path"] = selected_topics_df["topic_path"].astype("string")
            subtopic_lookup_df["topic_path"] = subtopic_lookup_df["topic_path"].astype("string")
            selected_topics_df = selected_topics_df.merge(
                subtopic_lookup_df,
                on="topic_path",
                how="left",
            )

        missing_topic_mask = selected_topics_df["dominant_topic"].isna()
        if missing_topic_mask.all():
            st.warning("No precomputed LDA topic assignments were found for the selected verses.")
        else:
            missing_references = selected_topics_df.loc[missing_topic_mask, "reference"].tolist()
            if missing_references:
                st.caption(f"Missing topic assignments for: {', '.join(missing_references)}")

            display_selected_topics_df = selected_topics_df.copy()
            display_selected_topics_df["dominant_topic"] = display_selected_topics_df["dominant_topic"].astype("Int64")
            display_selected_topics_df["Dominant Topic"] = display_selected_topics_df["dominant_topic"].map(
                lambda topic_id: f"Topic {topic_id}" if pd.notna(topic_id) else "N/A"
            )
            display_selected_topics_df["Confidence"] = display_selected_topics_df["topic_confidence"].map(
                lambda score: f"{score:.3f}" if pd.notna(score) else "N/A"
            )

            selected_topic_columns = ["reference", "Dominant Topic", "Confidence", "top_terms"]
            selected_topic_renames = {
                "reference": "Verse",
                "top_terms": "Topic Terms",
            }

            if has_subtopic_assignments:
                display_selected_topics_df["topic_path"] = display_selected_topics_df["topic_path"].fillna("N/A")
                display_selected_topics_df["Subtopic Confidence"] = display_selected_topics_df[
                    "subtopic_confidence"
                ].map(lambda score: f"{score:.3f}" if pd.notna(score) else "N/A")
                selected_topic_columns.extend(["topic_path", "Subtopic Confidence"])
                selected_topic_renames["topic_path"] = "Topic Path"

                if "subtopic_terms" in display_selected_topics_df.columns:
                    selected_topic_columns.append("subtopic_terms")
                    selected_topic_renames["subtopic_terms"] = "Subtopic Terms"

            st.dataframe(
                display_selected_topics_df[selected_topic_columns].rename(columns=selected_topic_renames),
                use_container_width=True,
                hide_index=True,
            )

            if has_subtopic_assignments:
                st.caption("Subtopic path confidence across selected verses")
                subtopic_confidence_df = display_selected_topics_df[
                    ["reference", "topic_path", "subtopic_confidence"]
                ].copy()
                subtopic_confidence_df = subtopic_confidence_df.dropna(subset=["subtopic_confidence"])

                if subtopic_confidence_df.empty:
                    st.info("No subtopic confidence scores are available for the selected verses.")
                else:
                    subtopic_confidence_chart = (
                        alt.Chart(subtopic_confidence_df)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X("reference:N", title="Verse"),
                            y=alt.Y("subtopic_confidence:Q", title="Subtopic Confidence"),
                            color=alt.Color("topic_path:N", title="Topic Path"),
                            tooltip=[
                                alt.Tooltip("reference:N", title="Verse"),
                                alt.Tooltip("topic_path:N", title="Topic Path"),
                                alt.Tooltip(
                                    "subtopic_confidence:Q",
                                    title="Confidence",
                                    format=".3f",
                                ),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(subtopic_confidence_chart, use_container_width=True)

                if has_subtopic_summary:
                    st.caption("Corpus subtopic prevalence")
                    subtopic_summary_view_df = subtopic_summary_df.copy()
                    subtopic_summary_view_df["topic_path_label"] = subtopic_summary_view_df.apply(
                        lambda row: topic_path_label(
                            row["topic_path"],
                            row.get("top_terms", ""),
                            max_terms=3,
                        ),
                        axis=1,
                    )

                    col_subtopic_overview, col_subtopic_terms = st.columns([1.5, 1])

                    with col_subtopic_overview:
                        subtopic_prevalence_chart = (
                            alt.Chart(subtopic_summary_view_df)
                            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                            .encode(
                                x=alt.X(
                                    "topic_path_label:N",
                                    title="Topic Path",
                                    axis=alt.Axis(labelAngle=-20, labelLimit=280),
                                ),
                                y=alt.Y("doc_count:Q", title="Verses"),
                                color=alt.Color(
                                    "avg_confidence:Q",
                                    title="Avg Confidence",
                                    scale=alt.Scale(scheme="tealblues"),
                                ),
                                tooltip=[
                                    alt.Tooltip("parent_topic:Q", title="Parent Topic"),
                                    alt.Tooltip("subtopic:Q", title="Subtopic"),
                                    alt.Tooltip("topic_path:N", title="Topic Path"),
                                    alt.Tooltip("doc_count:Q", title="Verse Count", format=","),
                                    alt.Tooltip("avg_confidence:Q", title="Avg Confidence", format=".3f"),
                                    alt.Tooltip("top_terms:N", title="Top Terms"),
                                ],
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(subtopic_prevalence_chart, use_container_width=True)

                    with col_subtopic_terms:
                        st.caption("Subtopic parts (top stems)")
                        parent_topic_options = sorted(
                            subtopic_summary_view_df["parent_topic"].unique().tolist()
                        )
                        selected_parent_topic = st.selectbox(
                            "Parent Topic",
                            parent_topic_options,
                            format_func=lambda topic_id: topic_label(
                                topic_id,
                                topic_summary_df.loc[
                                    topic_summary_df["topic"] == topic_id,
                                    "top_terms",
                                ].iloc[0],
                                max_terms=3,
                            ),
                            key="lda_parent_topic_select",
                        )

                        parent_subtopics_df = subtopic_summary_view_df[
                            subtopic_summary_view_df["parent_topic"] == selected_parent_topic
                        ].copy()

                        selected_topic_path = st.selectbox(
                            "Inspect Topic Path",
                            parent_subtopics_df["topic_path"].tolist(),
                            format_func=lambda path: topic_path_label(
                                path,
                                parent_subtopics_df.loc[
                                    parent_subtopics_df["topic_path"] == path,
                                    "top_terms",
                                ].iloc[0],
                                max_terms=3,
                            ),
                            key="lda_topic_path_select",
                        )

                        selected_topic_path_row = parent_subtopics_df[
                            parent_subtopics_df["topic_path"] == selected_topic_path
                        ].iloc[0]

                        if has_subtopic_term_weights:
                            selected_subtopic_terms = (
                                subtopic_term_weights_df[
                                    (subtopic_term_weights_df["parent_topic"] == int(selected_topic_path_row["parent_topic"]))
                                    & (subtopic_term_weights_df["subtopic"] == int(selected_topic_path_row["subtopic"]))
                                ]
                                .sort_values("rank")
                                .head(12)
                                .copy()
                            )
                        else:
                            selected_subtopic_terms = pd.DataFrame()

                        if selected_subtopic_terms.empty:
                            st.info("No subtopic term-weight rows are available for this topic path.")
                        else:
                            selected_subtopic_terms["term_rank_label"] = (
                                selected_subtopic_terms["rank"].astype(str)
                                + ". "
                                + selected_subtopic_terms["term"]
                            )

                            subtopic_terms_chart = (
                                alt.Chart(selected_subtopic_terms)
                                .mark_bar(cornerRadiusEnd=5)
                                .encode(
                                    x=alt.X("weight:Q", title="Weight"),
                                    y=alt.Y(
                                        "term_rank_label:N",
                                        sort=alt.SortField("rank", order="ascending"),
                                        title="Top Terms",
                                    ),
                                    color=alt.value("#2f7f5f"),
                                    tooltip=[
                                        alt.Tooltip("parent_topic:Q", title="Parent Topic"),
                                        alt.Tooltip("subtopic:Q", title="Subtopic"),
                                        alt.Tooltip("rank:Q", title="Rank"),
                                        alt.Tooltip("term:N", title="Term"),
                                        alt.Tooltip("weight:Q", title="Weight", format=".3f"),
                                    ],
                                )
                                .properties(height=320)
                            )
                            st.altair_chart(subtopic_terms_chart, use_container_width=True)
                else:
                    st.info(
                        "Subtopic assignments are present, but subtopic summary artifacts are missing. "
                        "Run `python build_lda_artifacts.py` to regenerate artifacts."
                    )

            topic_probability_columns = sorted(
                [
                    column
                    for column in display_selected_topics_df.columns
                    if column.startswith("topic_") and column[6:].isdigit()
                ],
                key=lambda column: int(column.split("_")[1]),
            )

            if topic_probability_columns:
                verse_topic_mix_df = display_selected_topics_df[
                    ["reference"] + topic_probability_columns
                ].melt(
                    id_vars="reference",
                    var_name="topic_column",
                    value_name="topic_probability",
                )
                verse_topic_mix_df["topic"] = (
                    verse_topic_mix_df["topic_column"].str.replace("topic_", "", regex=False).astype(int)
                )
                verse_topic_mix_df = verse_topic_mix_df.merge(
                    topic_summary_df[["topic", "top_terms"]],
                    on="topic",
                    how="left",
                )
                verse_topic_mix_df["topic_label"] = verse_topic_mix_df.apply(
                    lambda row: topic_label(row["topic"], row.get("top_terms", ""), max_terms=3),
                    axis=1,
                )

                mix_chart = (
                    alt.Chart(verse_topic_mix_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("reference:N", title="Verse"),
                        y=alt.Y(
                            "topic_probability:Q",
                            stack="normalize",
                            title="Topic Distribution",
                            axis=alt.Axis(format="%"),
                        ),
                        color=alt.Color("topic_label:N", title="Topic"),
                        tooltip=[
                            alt.Tooltip("reference:N", title="Verse"),
                            alt.Tooltip("topic:Q", title="Topic"),
                            alt.Tooltip("topic_probability:Q", title="Probability", format=".3f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(mix_chart, use_container_width=True)

# word level analysis

st.write("## Word-Level Analysis")

# Show word analysis for each selected verse in tabs
if selected_verses:
    word_tabs = st.tabs([f"{book} {ch}:{v}" for ch, v in selected_verses])
    
    for tab, (idx, (chapter, verse)) in zip(word_tabs, enumerate(selected_verses)):
        with tab:
            current_verse_df = current_verse_dfs[idx]
            if current_verse_df.empty:
                st.warning(f"No word data found for {book} {chapter}:{verse}")
            else:
                for i, (_, row) in enumerate(current_verse_df.iterrows()):
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        if st.button(row['text'], key=f"btn_{chapter}_{verse}_{i}"):
                            st.session_state.selected_greek = row['text']
                            st.session_state.selected_english = row['translation']
                            
                    with col2:
                        st.write(row['translation'])
                        
                    with col3:
                        st.write(row['transliteration']) 
                    
                    st.divider()
