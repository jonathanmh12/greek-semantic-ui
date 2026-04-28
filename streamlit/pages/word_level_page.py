import streamlit as st
import pandas as pd

import load_data_func

nt_split_column_df = load_data_func.load_verse_data()
translation_data = load_data_func.load_translation_data()
kjv_data = load_data_func.load_kjv_data()
books = sorted(translation_data['book'].dropna().unique())


# word level analysis

st.set_page_config(page_title="Word-Level Analysis", page_icon="🔍", layout="wide")
st.title("Word-Level Analysis of New Testament Verses")

with st.sidebar:
	st.title("Topic Modeling")
	book = st.selectbox("Select Book", books)

	book_mask = translation_data["book"] == book
	chapter_tokens = translation_data.loc[book_mask, "chapter"].dropna()
	chapter_list = sorted(chapter_tokens.unique().astype(int), key=int)

	st.subheader("Select Verse Range (up to 5 consecutive verses)")

	col1, col2 = st.columns(2)
	with col1:
		start_chapter = st.selectbox("Start Chapter", chapter_list, key="topic_start_ch")
	with col2:
		end_chapter = st.selectbox(
			"End Chapter",
			chapter_list,
			key="topic_end_ch",
			index=min(chapter_list.index(start_chapter), len(chapter_list) - 1),
		)

	start_chapter_mask = book_mask & (translation_data["chapter"] == start_chapter)
	start_verse_tokens = translation_data.loc[start_chapter_mask, "verse"].dropna()
	start_verse_list = sorted(
		start_verse_tokens[start_verse_tokens != ""].unique().astype(int),
		key=load_data_func.token_sort_key,
	)

	end_chapter_mask = book_mask & (translation_data["chapter"] == end_chapter)
	end_verse_tokens = translation_data.loc[end_chapter_mask, "verse"].dropna()
	end_verse_list = sorted(
		end_verse_tokens[end_verse_tokens != ""].unique().astype(int),
		key=load_data_func.token_sort_key,
	)

	col1, col2 = st.columns(2)
	with col1:
		start_verse = st.selectbox("Start Verse", start_verse_list, key="topic_start_vs")
	with col2:
		end_verse = st.selectbox("End Verse", end_verse_list, key="topic_end_vs")

	selected_verses = load_data_func.get_verse_range(
		book,
		start_chapter,
		start_verse,
		end_chapter,
		end_verse,
		translation_data,
	)

	if not selected_verses:
		st.error("Invalid verse range selection")
	else:
		st.success(f"Selected {len(selected_verses)} consecutive verse(s)")

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
