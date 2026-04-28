import streamlit as st

st.set_page_config(page_title="Greek Semantic UI", page_icon="📚", layout="wide")


def title_page():
	st.title("Greek Semantic UI")
	st.caption("Semantic exploration for New Testament Greek passages")

	st.markdown("---")

	st.subheader("Purpose")
	st.write(
		"This app helps you explore relationships between Greek New Testament verses "
		"using meaning-based search rather than only keyword matching. It is designed "
		"for study workflows where you want to quickly compare related passages, inspect "
		"themes, and move between source text and translation-level interpretation."
	)

	st.subheader("Technology (High Level)")
	st.write(
		"The application uses transformer-based sentence embeddings to encode verses as "
		"vectors in semantic space. A FAISS vector index powers fast nearest-neighbor "
		"retrieval, and the Streamlit interface provides an interactive way to navigate "
		"books, select verses, and inspect similarity-driven results. Caching is used to "
		"keep model loading and data access responsive during exploration."
	)


pg = st.navigation(
	[
		st.Page(title_page, title="Title Page", icon="🏠"),
		st.Page("pages/new_streamlit_ui.py", title="Verse Analysis", icon="📜"),
		st.Page("pages/topic_modeling.py", title="Topic Modeling", icon="🗂️"),
		st.Page("pages/word_level_page.py", title="Word-Level Analysis", icon="🔍"),
	]
)
pg.run()