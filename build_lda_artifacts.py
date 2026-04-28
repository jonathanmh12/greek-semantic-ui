from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

PROJECT_ROOT = Path(__file__).resolve().parent
LDA_DIR = PROJECT_ROOT / "data" / "greek-nt-sbert_v2" / "lda"
CORPUS_PATH = PROJECT_ROOT / "data" / "bible_corpus.csv"
TOP_TERMS_PER_TOPIC = 20
TOP_LABEL_TERMS = 12
TARGET_SUBTOPICS_PER_PARENT = 4
MIN_DOCS_TO_SPLIT = 120
MIN_DOCS_PER_SUBTOPIC = 60
MAX_SUBTOPICS_PER_PARENT = 6
SUBTOPIC_RANDOM_STATE = 42

RAW_GREEK_STOPWORDS = {
    "και", "δε", "γαρ", "ουν", "ως", "μη", "ου", "ουκ", "ουχ",
    "ο", "η", "το", "οι", "αι", "τα",
    "του", "της", "των", "τω", "τη", "τον", "την", "τους", "τας",
    "τοις", "ταις",
    "εν", "εις", "εκ", "εξ", "επι", "δια", "κατα", "προς", "υπο",
    "υπερ", "μετα", "παρα", "απο", "περι",
    "οτι", "ινα", "εαν", "αν",
    "τις", "τι", "τινα", "τινες",
    "αυτου", "αυτων", "αυτον", "αυτη", "αυτης", "αυται", "αυτοι", "αυτω",
    "ημιν", "ημων", "ημας", "υμιν", "υμων", "υμας",
    "μου", "σου", "με", "σε",
}

RAW_STEM_SUFFIXES = {
    "ομεθα", "ουμεθα", "εσθαι", "ησονται", "ονται", "ουσιν", "ουσι",
    "ομεν", "ουμεν", "ειτε", "ετε", "εται", "ειται", "ειν", "εις",
    "οντες", "οντας", "μενος", "μενη", "μενον", "ματος", "ματων",
    "μασι", "σεως", "σεων", "σεσι", "τητες", "τητων", "τητα",
    "ικος", "ικου", "ικον", "ικοι", "ικων",
    "οις", "αις", "ους", "ων", "ου", "ος", "ον", "οι", "αι",
    "ας", "ες", "ης", "ην", "ει",
}


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


GREEK_STOPWORDS = {
    strip_diacritics(word.lower()).replace("ς", "σ") for word in RAW_GREEK_STOPWORDS
}

STEM_SUFFIXES = sorted(
    {suffix.replace("ς", "σ") for suffix in RAW_STEM_SUFFIXES},
    key=len,
    reverse=True,
)


def normalize_greek(text: str) -> str:
    text = strip_diacritics(str(text).lower()).replace("ς", "σ")
    text = re.sub(r"[^α-ω\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def light_stem_greek_token(token: str) -> str:
    if len(token) <= 4:
        return token

    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]

    return token


def preprocess_to_corpus_text(text: str) -> str:
    normalized = normalize_greek(text)
    tokens = [
        token
        for token in normalized.split()
        if len(token) > 2 and token not in GREEK_STOPWORDS
    ]
    stems = [light_stem_greek_token(token) for token in tokens]
    stems = [stem for stem in stems if len(stem) > 1]
    return " ".join(stems)


def ensure_inputs_exist() -> None:
    required_paths = [
        LDA_DIR / "count_vectorizer.pkl",
        LDA_DIR / "lda_model.pkl",
        CORPUS_PATH,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")


def describe_topics(lda_model, feature_names: np.ndarray, n_words: int = TOP_LABEL_TERMS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topic_index, weights in enumerate(lda_model.components_):
        top_indices = weights.argsort()[-n_words:][::-1]
        rows.append(
            {
                "topic": int(topic_index),
                "top_terms": ", ".join(str(feature_names[i]) for i in top_indices),
            }
        )
    return pd.DataFrame(rows)


def build_subtopic_views(
    verse_topics_df: pd.DataFrame,
    doc_term_matrix,
    feature_names: np.ndarray,
    lda_model,
    topic_labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parent_topic_terms = topic_labels_df.set_index("topic")["top_terms"].to_dict()
    topic_column = "dominant_topic"
    confidence_column = "topic_confidence"

    verse_topics_df["subtopic"] = 0
    verse_topics_df["subtopic_confidence"] = verse_topics_df[confidence_column]
    verse_topics_df["topic_path"] = verse_topics_df[topic_column].astype(int).astype(str) + ".0"

    subtopic_summary_rows: list[dict[str, object]] = []
    subtopic_term_rows: list[dict[str, object]] = []

    for parent_topic in sorted(verse_topics_df[topic_column].astype(int).unique()):
        mask = verse_topics_df[topic_column].astype(int) == int(parent_topic)
        parent_indices = np.flatnonzero(mask.to_numpy())
        n_docs = int(parent_indices.size)
        parent_topic_label_terms = parent_topic_terms.get(int(parent_topic), "")

        if n_docs < MIN_DOCS_TO_SPLIT:
            parent_weights = lda_model.components_[int(parent_topic)]
            top_indices = parent_weights.argsort()[::-1][:TOP_TERMS_PER_TOPIC]
            for rank, term_idx in enumerate(top_indices, start=1):
                subtopic_term_rows.append(
                    {
                        "parent_topic": int(parent_topic),
                        "subtopic": 0,
                        "topic_path": f"{int(parent_topic)}.0",
                        "rank": int(rank),
                        "term": str(feature_names[term_idx]),
                        "weight": float(parent_weights[term_idx]),
                        "split_applied": False,
                    }
                )

            subtopic_summary_rows.append(
                {
                    "parent_topic": int(parent_topic),
                    "subtopic": 0,
                    "topic_path": f"{int(parent_topic)}.0",
                    "doc_count": n_docs,
                    "avg_confidence": float(verse_topics_df.loc[mask, confidence_column].mean()),
                    "top_terms": parent_topic_label_terms,
                    "split_applied": False,
                }
            )
            continue

        k_by_size = max(2, n_docs // MIN_DOCS_PER_SUBTOPIC)
        n_subtopics = min(MAX_SUBTOPICS_PER_PARENT, TARGET_SUBTOPICS_PER_PARENT, k_by_size)
        n_subtopics = min(n_subtopics, max(2, n_docs - 1))

        parent_doc_term_matrix = doc_term_matrix[parent_indices]
        sub_lda = LatentDirichletAllocation(
            n_components=n_subtopics,
            learning_method="batch",
            max_iter=15,
            random_state=SUBTOPIC_RANDOM_STATE,
        )
        sub_doc_topic = sub_lda.fit_transform(parent_doc_term_matrix)

        sub_ids = sub_doc_topic.argmax(axis=1)
        sub_conf = sub_doc_topic.max(axis=1)

        verse_topics_df.loc[mask, "subtopic"] = sub_ids
        verse_topics_df.loc[mask, "subtopic_confidence"] = sub_conf
        verse_topics_df.loc[mask, "topic_path"] = [
            f"{int(parent_topic)}.{int(sub_id)}" for sub_id in sub_ids
        ]

        subtopic_terms_df = describe_topics(sub_lda, feature_names, n_words=10)
        for row in subtopic_terms_df.itertuples(index=False):
            sub_mask = sub_ids == row.topic
            if not sub_mask.any():
                continue

            subtopic_summary_rows.append(
                {
                    "parent_topic": int(parent_topic),
                    "subtopic": int(row.topic),
                    "topic_path": f"{int(parent_topic)}.{int(row.topic)}",
                    "doc_count": int(sub_mask.sum()),
                    "avg_confidence": float(sub_conf[sub_mask].mean()),
                    "top_terms": row.top_terms,
                    "split_applied": True,
                }
            )

        for subtopic_idx, sub_weights in enumerate(sub_lda.components_):
            top_indices = sub_weights.argsort()[::-1][:TOP_TERMS_PER_TOPIC]
            for rank, term_idx in enumerate(top_indices, start=1):
                subtopic_term_rows.append(
                    {
                        "parent_topic": int(parent_topic),
                        "subtopic": int(subtopic_idx),
                        "topic_path": f"{int(parent_topic)}.{int(subtopic_idx)}",
                        "rank": int(rank),
                        "term": str(feature_names[term_idx]),
                        "weight": float(sub_weights[term_idx]),
                        "split_applied": True,
                    }
                )

    subtopic_summary_df = pd.DataFrame(subtopic_summary_rows)
    if not subtopic_summary_df.empty:
        subtopic_summary_df = (
            subtopic_summary_df.sort_values(["parent_topic", "subtopic"]).reset_index(drop=True)
        )
        subtopic_summary_df["avg_confidence"] = subtopic_summary_df["avg_confidence"].round(6)

    subtopic_term_weights_df = pd.DataFrame(subtopic_term_rows)
    if not subtopic_term_weights_df.empty:
        subtopic_term_weights_df = (
            subtopic_term_weights_df.sort_values(["parent_topic", "subtopic", "rank"])
            .reset_index(drop=True)
        )

    return verse_topics_df, subtopic_summary_df, subtopic_term_weights_df


def build_artifacts() -> None:
    ensure_inputs_exist()

    vectorizer = joblib.load(LDA_DIR / "count_vectorizer.pkl")
    lda_model = joblib.load(LDA_DIR / "lda_model.pkl")
    corpus_df = pd.read_csv(CORPUS_PATH)

    if "verse_ref" not in corpus_df.columns or "text" not in corpus_df.columns:
        raise ValueError("Expected columns `verse_ref` and `text` in data/bible_corpus.csv")

    if "corpus_text" in corpus_df.columns:
        corpus_text = corpus_df["corpus_text"].fillna("").astype(str)
    else:
        corpus_text = corpus_df["text"].fillna("").map(preprocess_to_corpus_text)

    doc_term_matrix = vectorizer.transform(corpus_text)
    doc_topic_matrix = lda_model.transform(doc_term_matrix)

    topic_columns = [f"topic_{topic_idx}" for topic_idx in range(doc_topic_matrix.shape[1])]
    verse_topics_df = corpus_df[["verse_ref"]].copy()

    for topic_idx, topic_column in enumerate(topic_columns):
        verse_topics_df[topic_column] = doc_topic_matrix[:, topic_idx]

    verse_topics_df["dominant_topic"] = doc_topic_matrix.argmax(axis=1).astype(int)
    verse_topics_df["topic_confidence"] = doc_topic_matrix.max(axis=1)
    feature_names = vectorizer.get_feature_names_out()
    topic_labels_path = LDA_DIR / "topic_labels.csv"
    if topic_labels_path.exists():
        topic_labels_df = pd.read_csv(topic_labels_path)
    else:
        topic_labels_df = describe_topics(lda_model, feature_names)
        topic_labels_df.to_csv(topic_labels_path, index=False)

    verse_topics_df, subtopic_summary_df, subtopic_term_weights_df = build_subtopic_views(
        verse_topics_df=verse_topics_df,
        doc_term_matrix=doc_term_matrix,
        feature_names=feature_names,
        lda_model=lda_model,
        topic_labels_df=topic_labels_df,
    )

    confidence_columns = ["topic_confidence", "subtopic_confidence"]
    verse_topics_df[topic_columns + confidence_columns] = verse_topics_df[
        topic_columns + confidence_columns
    ].round(6)
    verse_topics_df["subtopic"] = verse_topics_df["subtopic"].astype(int)

    topic_counts_df = (
        verse_topics_df["dominant_topic"]
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="verse_count")
    )
    topic_conf_df = (
        verse_topics_df.groupby("dominant_topic", as_index=False)["topic_confidence"]
        .mean()
        .rename(columns={"dominant_topic": "topic", "topic_confidence": "avg_confidence"})
    )

    topic_summary_df = topic_counts_df.merge(topic_conf_df, on="topic", how="left")
    topic_summary_df["verse_share"] = topic_summary_df["verse_count"] / len(verse_topics_df)
    topic_summary_df = topic_summary_df.merge(topic_labels_df, on="topic", how="left").sort_values("topic")
    topic_summary_df[["avg_confidence", "verse_share"]] = topic_summary_df[["avg_confidence", "verse_share"]].round(6)

    term_rows: list[dict[str, object]] = []
    for topic_idx, weights in enumerate(lda_model.components_):
        top_indices = weights.argsort()[::-1][:TOP_TERMS_PER_TOPIC]
        for rank, term_idx in enumerate(top_indices, start=1):
            term_rows.append(
                {
                    "topic": int(topic_idx),
                    "rank": int(rank),
                    "term": str(feature_names[term_idx]),
                    "weight": float(weights[term_idx]),
                }
            )

    topic_term_weights_df = pd.DataFrame(term_rows)

    verse_topics_path = LDA_DIR / "verse_topics.csv"
    topic_summary_path = LDA_DIR / "topic_summary.csv"
    topic_term_weights_path = LDA_DIR / "topic_term_weights.csv"
    subtopic_summary_path = LDA_DIR / "subtopic_summary.csv"
    subtopic_term_weights_path = LDA_DIR / "subtopic_term_weights.csv"
    manifest_path = LDA_DIR / "artifacts_manifest.json"

    verse_topics_df.to_csv(verse_topics_path, index=False)
    topic_summary_df.to_csv(topic_summary_path, index=False)
    topic_term_weights_df.to_csv(topic_term_weights_path, index=False)
    subtopic_summary_df.to_csv(subtopic_summary_path, index=False)
    subtopic_term_weights_df.to_csv(subtopic_term_weights_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "corpus": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
            "vectorizer": str((LDA_DIR / "count_vectorizer.pkl").relative_to(PROJECT_ROOT)),
            "lda_model": str((LDA_DIR / "lda_model.pkl").relative_to(PROJECT_ROOT)),
        },
        "outputs": {
            "verse_topics": str(verse_topics_path.relative_to(PROJECT_ROOT)),
            "topic_summary": str(topic_summary_path.relative_to(PROJECT_ROOT)),
            "topic_term_weights": str(topic_term_weights_path.relative_to(PROJECT_ROOT)),
            "subtopic_summary": str(subtopic_summary_path.relative_to(PROJECT_ROOT)),
            "subtopic_term_weights": str(subtopic_term_weights_path.relative_to(PROJECT_ROOT)),
        },
        "n_verses": int(len(verse_topics_df)),
        "n_topics": int(lda_model.n_components),
        "n_topic_paths": int(subtopic_summary_df["topic_path"].nunique()),
        "n_features": int(len(feature_names)),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved {verse_topics_path}")
    print(f"Saved {topic_summary_path}")
    print(f"Saved {topic_term_weights_path}")
    print(f"Saved {subtopic_summary_path}")
    print(f"Saved {subtopic_term_weights_path}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    build_artifacts()
