import streamlit as st
import pandas as pd
import altair as alt
import re
from pathlib import Path

st.set_page_config(page_title="Topic Modeling", layout="wide")


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
	return pd.read_csv("data/og_lang_transformed.csv")


def token_sort_key(token):
	token = str(token)
	numeric = "".join(ch for ch in token if ch.isdigit())
	if numeric:
		return int(numeric), token
	return float("inf"), token


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
	book_chapters = sorted(data[data["book"] == book_name]["chapter"].dropna().unique().astype(int))

	while len(verses) < 5:
		chapter_verses = sorted(
			data[(data["book"] == book_name) & (data["chapter"] == current_chapter)]["verse"]
			.dropna()[lambda x: x != ""].unique().astype(int),
			key=int,
		)

		if not chapter_verses:
			break

		if current_chapter == start_chapter and current_chapter == end_chapter:
			for verse_number in chapter_verses:
				if start_verse <= verse_number <= end_verse and len(verses) < 5:
					verses.append((current_chapter, verse_number))
			break
		if current_chapter == start_chapter:
			for verse_number in chapter_verses:
				if verse_number >= start_verse and len(verses) < 5:
					verses.append((current_chapter, verse_number))
		elif current_chapter == end_chapter:
			for verse_number in chapter_verses:
				if verse_number <= end_verse and len(verses) < 5:
					verses.append((current_chapter, verse_number))
			break
		else:
			for verse_number in chapter_verses:
				if len(verses) < 5:
					verses.append((current_chapter, verse_number))

		chapter_idx = book_chapters.index(current_chapter)
		if chapter_idx + 1 < len(book_chapters):
			current_chapter = book_chapters[chapter_idx + 1]
		else:
			break

	return verses


translation_data = load_translation_data()
lda_artifacts = load_lda_artifacts()
books = sorted(translation_data["book"].dropna().unique())

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
		key=token_sort_key,
	)

	end_chapter_mask = book_mask & (translation_data["chapter"] == end_chapter)
	end_verse_tokens = translation_data.loc[end_chapter_mask, "verse"].dropna()
	end_verse_list = sorted(
		end_verse_tokens[end_verse_tokens != ""].unique().astype(int),
		key=token_sort_key,
	)

	col1, col2 = st.columns(2)
	with col1:
		start_verse = st.selectbox("Start Verse", start_verse_list, key="topic_start_vs")
	with col2:
		end_verse = st.selectbox("End Verse", end_verse_list, key="topic_end_vs")

	selected_verses = get_verse_range(
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

st.title("LDA Topic Insights")

# if not selected_verses:
# 	st.warning("Select a valid verse range in the sidebar.")
# elif lda_artifacts is None:
# 	st.info("Precomputed LDA artifacts are missing. Run `python build_lda_artifacts.py` once to generate them.")
# else:
# 	topic_summary_df = lda_artifacts["topic_summary"].copy()
# 	topic_term_weights_df = lda_artifacts["topic_term_weights"].copy()
# 	verse_topics_df = lda_artifacts["verse_topics"].copy()
# 	subtopic_summary_df = lda_artifacts.get("subtopic_summary", pd.DataFrame()).copy()
# 	subtopic_term_weights_df = lda_artifacts.get("subtopic_term_weights", pd.DataFrame()).copy()

# 	has_subtopic_assignments = {"subtopic", "subtopic_confidence", "topic_path"}.issubset(
# 		verse_topics_df.columns
# 	)
# 	has_subtopic_summary = not subtopic_summary_df.empty
# 	has_subtopic_term_weights = not subtopic_term_weights_df.empty

# 	topic_summary_df["topic_label"] = topic_summary_df.apply(
# 		lambda row: topic_label(row["topic"], row.get("top_terms", "")),
# 		axis=1,
# 	)

# 	col_overview, col_topic_terms = st.columns([1.5, 1])

# 	with col_overview:
# 		st.caption("Corpus topic prevalence")
# 		prevalence_chart = (
# 			alt.Chart(topic_summary_df)
# 			.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
# 			.encode(
# 				x=alt.X(
# 					"topic_label:N",
# 					title="Topic",
# 					axis=alt.Axis(labelAngle=-20, labelLimit=260),
# 				),
# 				y=alt.Y("verse_count:Q", title="Verses"),
# 				color=alt.Color(
# 					"avg_confidence:Q",
# 					title="Avg Confidence",
# 					scale=alt.Scale(scheme="tealblues"),
# 				),
# 				tooltip=[
# 					alt.Tooltip("topic:Q", title="Topic"),
# 					alt.Tooltip("verse_count:Q", title="Verse Count", format=","),
# 					alt.Tooltip("verse_share:Q", title="Corpus Share", format=".2%"),
# 					alt.Tooltip("avg_confidence:Q", title="Avg Confidence", format=".3f"),
# 					alt.Tooltip("top_terms:N", title="Top Terms"),
# 				],
# 			)
# 			.properties(height=280)
# 		)
# 		st.altair_chart(prevalence_chart, use_container_width=True)

# 	with col_topic_terms:
# 		st.caption("Topic parts (top stems)")
# 		topic_options = topic_summary_df["topic"].tolist()
# 		selected_topic = st.selectbox(
# 			"Inspect Topic",
# 			topic_options,
# 			format_func=lambda topic_id: topic_label(
# 				topic_id,
# 				topic_summary_df.loc[topic_summary_df["topic"] == topic_id, "top_terms"].iloc[0],
# 				max_terms=3,
# 			),
# 			key="topic_modeling_lda_topic_select",
# 		)

# 		selected_term_weights = (
# 			topic_term_weights_df[topic_term_weights_df["topic"] == selected_topic]
# 			.sort_values("rank")
# 			.head(12)
# 			.copy()
# 		)
# 		selected_term_weights["term_rank_label"] = (
# 			selected_term_weights["rank"].astype(str) + ". " + selected_term_weights["term"]
# 		)

# 		terms_chart = (
# 			alt.Chart(selected_term_weights)
# 			.mark_bar(cornerRadiusEnd=5)
# 			.encode(
# 				x=alt.X("weight:Q", title="Weight"),
# 				y=alt.Y(
# 					"term_rank_label:N",
# 					sort=alt.SortField("rank", order="ascending"),
# 					title="Top Terms",
# 				),
# 				color=alt.value("#2f7f5f"),
# 				tooltip=[
# 					alt.Tooltip("topic:Q", title="Topic"),
# 					alt.Tooltip("rank:Q", title="Rank"),
# 					alt.Tooltip("term:N", title="Term"),
# 					alt.Tooltip("weight:Q", title="Weight", format=".3f"),
# 				],
# 			)
# 			.properties(height=320)
# 		)
# 		st.altair_chart(terms_chart, use_container_width=True)

# 	st.caption("Dominant topic per selected verse")
# 	selected_ref_rows = [
# 		{
# 			"reference": f"{book} {chapter}:{verse}",
# 			"verse_ref_key": to_verse_ref_key(book, chapter, verse),
# 		}
# 		for chapter, verse in selected_verses
# 	]
# 	selected_ref_df = pd.DataFrame(selected_ref_rows)

# 	selected_topics_df = selected_ref_df.merge(
# 		verse_topics_df,
# 		on="verse_ref_key",
# 		how="left",
# 	)
# 	selected_topics_df = selected_topics_df.merge(
# 		topic_summary_df[["topic", "top_terms"]],
# 		left_on="dominant_topic",
# 		right_on="topic",
# 		how="left",
# 	)

# 	if has_subtopic_assignments and has_subtopic_summary:
# 		subtopic_lookup_df = (
# 			subtopic_summary_df[["topic_path", "top_terms"]]
# 			.rename(columns={"top_terms": "subtopic_terms"})
# 			.drop_duplicates(subset=["topic_path"])
# 		)
# 		selected_topics_df["topic_path"] = selected_topics_df["topic_path"].astype("string")
# 		subtopic_lookup_df["topic_path"] = subtopic_lookup_df["topic_path"].astype("string")
# 		selected_topics_df = selected_topics_df.merge(
# 			subtopic_lookup_df,
# 			on="topic_path",
# 			how="left",
# 		)

# 	missing_topic_mask = selected_topics_df["dominant_topic"].isna()
# 	if missing_topic_mask.all():
# 		st.warning("No precomputed LDA topic assignments were found for the selected verses.")
# 	else:
# 		missing_references = selected_topics_df.loc[missing_topic_mask, "reference"].tolist()
# 		if missing_references:
# 			st.caption(f"Missing topic assignments for: {', '.join(missing_references)}")

# 		display_selected_topics_df = selected_topics_df.copy()
# 		display_selected_topics_df["dominant_topic"] = display_selected_topics_df["dominant_topic"].astype("Int64")
# 		display_selected_topics_df["Dominant Topic"] = display_selected_topics_df["dominant_topic"].map(
# 			lambda topic_id: f"Topic {topic_id}" if pd.notna(topic_id) else "N/A"
# 		)
# 		display_selected_topics_df["Confidence"] = display_selected_topics_df["topic_confidence"].map(
# 			lambda score: f"{score:.3f}" if pd.notna(score) else "N/A"
# 		)

# 		selected_topic_columns = ["reference", "Dominant Topic", "Confidence", "top_terms"]
# 		selected_topic_renames = {
# 			"reference": "Verse",
# 			"top_terms": "Topic Terms",
# 		}

# 		if has_subtopic_assignments:
# 			display_selected_topics_df["topic_path"] = display_selected_topics_df["topic_path"].fillna("N/A")
# 			display_selected_topics_df["Subtopic Confidence"] = display_selected_topics_df[
# 				"subtopic_confidence"
# 			].map(lambda score: f"{score:.3f}" if pd.notna(score) else "N/A")
# 			selected_topic_columns.extend(["topic_path", "Subtopic Confidence"])
# 			selected_topic_renames["topic_path"] = "Topic Path"

# 			if "subtopic_terms" in display_selected_topics_df.columns:
# 				selected_topic_columns.append("subtopic_terms")
# 				selected_topic_renames["subtopic_terms"] = "Subtopic Terms"

# 		st.dataframe(
# 			display_selected_topics_df[selected_topic_columns].rename(columns=selected_topic_renames),
# 			use_container_width=True,
# 			hide_index=True,
# 		)

# 		if has_subtopic_assignments:
# 			st.caption("Subtopic path confidence across selected verses")
# 			subtopic_confidence_df = display_selected_topics_df[
# 				["reference", "topic_path", "subtopic_confidence"]
# 			].copy()
# 			subtopic_confidence_df = subtopic_confidence_df.dropna(subset=["subtopic_confidence"])

# 			if subtopic_confidence_df.empty:
# 				st.info("No subtopic confidence scores are available for the selected verses.")
# 			else:
# 				subtopic_confidence_chart = (
# 					alt.Chart(subtopic_confidence_df)
# 					.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
# 					.encode(
# 						x=alt.X("reference:N", title="Verse"),
# 						y=alt.Y("subtopic_confidence:Q", title="Subtopic Confidence"),
# 						color=alt.Color("topic_path:N", title="Topic Path"),
# 						tooltip=[
# 							alt.Tooltip("reference:N", title="Verse"),
# 							alt.Tooltip("topic_path:N", title="Topic Path"),
# 							alt.Tooltip(
# 								"subtopic_confidence:Q",
# 								title="Confidence",
# 								format=".3f",
# 							),
# 						],
# 					)
# 					.properties(height=220)
# 				)
# 				st.altair_chart(subtopic_confidence_chart, use_container_width=True)

# 			if has_subtopic_summary:
# 				st.caption("Corpus subtopic prevalence")
# 				subtopic_summary_view_df = subtopic_summary_df.copy()
# 				subtopic_summary_view_df["topic_path_label"] = subtopic_summary_view_df.apply(
# 					lambda row: topic_path_label(
# 						row["topic_path"],
# 						row.get("top_terms", ""),
# 						max_terms=3,
# 					),
# 					axis=1,
# 				)

# 				col_subtopic_overview, col_subtopic_terms = st.columns([1.5, 1])

# 				with col_subtopic_overview:
# 					subtopic_prevalence_chart = (
# 						alt.Chart(subtopic_summary_view_df)
# 						.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
# 						.encode(
# 							x=alt.X(
# 								"topic_path_label:N",
# 								title="Topic Path",
# 								axis=alt.Axis(labelAngle=-20, labelLimit=280),
# 							),
# 							y=alt.Y("doc_count:Q", title="Verses"),
# 							color=alt.Color(
# 								"avg_confidence:Q",
# 								title="Avg Confidence",
# 								scale=alt.Scale(scheme="tealblues"),
# 							),
# 							tooltip=[
# 								alt.Tooltip("parent_topic:Q", title="Parent Topic"),
# 								alt.Tooltip("subtopic:Q", title="Subtopic"),
# 								alt.Tooltip("topic_path:N", title="Topic Path"),
# 								alt.Tooltip("doc_count:Q", title="Verse Count", format=","),
# 								alt.Tooltip("avg_confidence:Q", title="Avg Confidence", format=".3f"),
# 								alt.Tooltip("top_terms:N", title="Top Terms"),
# 							],
# 						)
# 						.properties(height=280)
# 					)
# 					st.altair_chart(subtopic_prevalence_chart, use_container_width=True)

# 				with col_subtopic_terms:
# 					st.caption("Subtopic parts (top stems)")
# 					parent_topic_options = sorted(
# 						subtopic_summary_view_df["parent_topic"].unique().tolist()
# 					)
# 					selected_parent_topic = st.selectbox(
# 						"Parent Topic",
# 						parent_topic_options,
# 						format_func=lambda topic_id: topic_label(
# 							topic_id,
# 							topic_summary_df.loc[
# 								topic_summary_df["topic"] == topic_id,
# 								"top_terms",
# 							].iloc[0],
# 							max_terms=3,
# 						),
# 						key="topic_modeling_lda_parent_topic_select",
# 					)

# 					parent_subtopics_df = subtopic_summary_view_df[
# 						subtopic_summary_view_df["parent_topic"] == selected_parent_topic
# 					].copy()

# 					selected_topic_path = st.selectbox(
# 						"Inspect Topic Path",
# 						parent_subtopics_df["topic_path"].tolist(),
# 						format_func=lambda path: topic_path_label(
# 							path,
# 							parent_subtopics_df.loc[
# 								parent_subtopics_df["topic_path"] == path,
# 								"top_terms",
# 							].iloc[0],
# 							max_terms=3,
# 						),
# 						key="topic_modeling_lda_topic_path_select",
# 					)

# 					selected_topic_path_row = parent_subtopics_df[
# 						parent_subtopics_df["topic_path"] == selected_topic_path
# 					].iloc[0]

# 					if has_subtopic_term_weights:
# 						selected_subtopic_terms = (
# 							subtopic_term_weights_df[
# 								(subtopic_term_weights_df["parent_topic"] == int(selected_topic_path_row["parent_topic"]))
# 								& (subtopic_term_weights_df["subtopic"] == int(selected_topic_path_row["subtopic"]))
# 							]
# 							.sort_values("rank")
# 							.head(12)
# 							.copy()
# 						)
# 					else:
# 						selected_subtopic_terms = pd.DataFrame()

# 					if selected_subtopic_terms.empty:
# 						st.info("No subtopic term-weight rows are available for this topic path.")
# 					else:
# 						selected_subtopic_terms["term_rank_label"] = (
# 							selected_subtopic_terms["rank"].astype(str)
# 							+ ". "
# 							+ selected_subtopic_terms["term"]
# 						)

# 						subtopic_terms_chart = (
# 							alt.Chart(selected_subtopic_terms)
# 							.mark_bar(cornerRadiusEnd=5)
# 							.encode(
# 								x=alt.X("weight:Q", title="Weight"),
# 								y=alt.Y(
# 									"term_rank_label:N",
# 									sort=alt.SortField("rank", order="ascending"),
# 									title="Top Terms",
# 								),
# 								color=alt.value("#2f7f5f"),
# 								tooltip=[
# 									alt.Tooltip("parent_topic:Q", title="Parent Topic"),
# 									alt.Tooltip("subtopic:Q", title="Subtopic"),
# 									alt.Tooltip("rank:Q", title="Rank"),
# 									alt.Tooltip("term:N", title="Term"),
# 									alt.Tooltip("weight:Q", title="Weight", format=".3f"),
# 								],
# 							)
# 							.properties(height=320)
# 						)
# 						st.altair_chart(subtopic_terms_chart, use_container_width=True)
# 			else:
# 				st.info(
# 					"Subtopic assignments are present, but subtopic summary artifacts are missing. "
# 					"Run `python build_lda_artifacts.py` to regenerate artifacts."
# 				)

# 		topic_probability_columns = sorted(
# 			[
# 				column
# 				for column in display_selected_topics_df.columns
# 				if column.startswith("topic_") and column[6:].isdigit()
# 			],
# 			key=lambda column: int(column.split("_")[1]),
# 		)

# 		if topic_probability_columns:
# 			verse_topic_mix_df = display_selected_topics_df[
# 				["reference"] + topic_probability_columns
# 			].melt(
# 				id_vars="reference",
# 				var_name="topic_column",
# 				value_name="topic_probability",
# 			)
# 			verse_topic_mix_df["topic"] = (
# 				verse_topic_mix_df["topic_column"].str.replace("topic_", "", regex=False).astype(int)
# 			)
# 			verse_topic_mix_df = verse_topic_mix_df.merge(
# 				topic_summary_df[["topic", "top_terms"]],
# 				on="topic",
# 				how="left",
# 			)
# 			verse_topic_mix_df["topic_label"] = verse_topic_mix_df.apply(
# 				lambda row: topic_label(row["topic"], row.get("top_terms", ""), max_terms=3),
# 				axis=1,
# 			)

# 			mix_chart = (
# 				alt.Chart(verse_topic_mix_df)
# 				.mark_bar()
# 				.encode(
# 					x=alt.X("reference:N", title="Verse"),
# 					y=alt.Y(
# 						"topic_probability:Q",
# 						stack="normalize",
# 						title="Topic Distribution",
# 						axis=alt.Axis(format="%"),
# 					),
# 					color=alt.Color("topic_label:N", title="Topic"),
# 					tooltip=[
# 						alt.Tooltip("reference:N", title="Verse"),
# 						alt.Tooltip("topic:Q", title="Topic"),
# 						alt.Tooltip("topic_probability:Q", title="Probability", format=".3f"),
# 					],
# 				)
# 				.properties(height=260)
# 			)
# 			st.altair_chart(mix_chart, use_container_width=True)

