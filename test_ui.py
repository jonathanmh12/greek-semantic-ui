import os
import re

import numpy as np
import pygame
import torch

# Sample data for John 1:1 (expand as needed)
verse_data = [
    {"greek": "ἐν ἀρχῇ", "english": "In the beginning"},
    {"greek": "ἦν", "english": "was"},
    {"greek": "ὁ λόγος", "english": "the Word"},
    {"greek": "καὶ", "english": "and"},
    {"greek": "ὁ λόγος", "english": "the Word"},
    {"greek": "ἦν", "english": "was"},
    {"greek": "πρὸς τὸν θεόν", "english": "with God"},
    {"greek": "καὶ", "english": "and"},
    {"greek": "θεὸς", "english": "God"},
    {"greek": "ἦν", "english": "was"},
    {"greek": "ὁ λόγος", "english": "the Word"}
]

from sentence_transformers import SentenceTransformer, models

model_dir = "models/GreekBERT_v3"
corpus_path = "greek_corpus.txt"


def extract_words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)


def is_greek_word(word: str) -> bool:
    for char in word:
        code_point = ord(char)
        if (
            0x0370 <= code_point <= 0x03FF
            or 0x1F00 <= code_point <= 0x1FFF
            or 0x10140 <= code_point <= 0x1018F
        ):
            return True
    return False


def load_candidate_words(base_data: list[dict], corpus_file: str) -> list[str]:
    candidates: set[str] = set()

    for item in base_data:
        for token in extract_words(item["greek"]):
            if is_greek_word(token):
                candidates.add(token)

    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                for token in extract_words(line):
                    if is_greek_word(token):
                        candidates.add(token)
    else:
        print(
            f"Warning: {corpus_file} not found. "
            "Using only words from verse_data as candidates."
        )

    return sorted(candidates)


def build_embedding_index(texts: list[str]) -> dict[str, np.ndarray]:
    vectors = model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 500,
    )
    return {text: vector for text, vector in zip(texts, vectors)}


def get_embedding(text: str) -> np.ndarray:
    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

# Load the underlying transformer (your RoBERTa)
transformer = models.Transformer(
    model_name_or_path=model_dir,
    tokenizer_args={"use_fast": True},
    model_args={"trust_remote_code": False},  # usually not needed
)

# Add a pooling layer on top (mean pooling is usually best for similarity)
pooling_model = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode='mean'          # or 'cls' if you prefer CLS pooling
    # You can also add: pooling_mode_cls_token=True, etc. — experiment later
)

# Combine into a full SentenceTransformer
model = SentenceTransformer(modules=[transformer, pooling_model])

# Optional: move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Build candidate vocabulary from corpus + visible verse text
candidate_words = load_candidate_words(verse_data, corpus_path)
candidate_embeddings = build_embedding_index(candidate_words)

# Cache embeddings for displayed verse phrases
display_embeddings = build_embedding_index([item["greek"] for item in verse_data])

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pygame setup (same as before)
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Interlinear Verse UI with Embeddings")
font = pygame.font.SysFont("arial", 20)
small_font = pygame.font.SysFont("arial", 16)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Positions and widths
start_x = 50
start_y = 100
item_width = 120
separator_width = 10

# Store rects for clickable Greek items
greek_rects = []

# Selected Greek and similarities
selected_greek = None
similarities = []

running = True
while running:
    screen.fill(WHITE)
    
    # Draw English row
    x = start_x
    for item in verse_data:
        english_text = font.render(item["english"], True, BLACK)
        screen.blit(english_text, (x, start_y))
        x += item_width
    
    # Draw Greek row with separators and make clickable
    x = start_x
    greek_rects = []  # Reset rects
    for i, item in enumerate(verse_data):
        if i > 0:
            pygame.draw.line(screen, GRAY, (x - separator_width // 2, start_y + 40), (x - separator_width // 2, start_y + 80), 2)
        
        greek_text = font.render(item["greek"], True, BLUE)
        rect = greek_text.get_rect(topleft=(x, start_y + 50))
        screen.blit(greek_text, rect)
        greek_rects.append((rect, item["greek"]))
        x += item_width
    
    # Draw similarity options if selected
    if selected_greek:
        similarities = []
        selected_emb = display_embeddings.get(selected_greek, get_embedding(selected_greek))
        for other_greek, other_emb in candidate_embeddings.items():
            if other_greek != selected_greek:
                sim = cosine_similarity(selected_emb, other_emb)
                similarities.append((other_greek, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        sidebar_x = 50
        sidebar_y = 200
        screen.blit(font.render(f"Selected: {selected_greek}", True, BLACK), (sidebar_x, sidebar_y))
        sidebar_y += 40
        screen.blit(
            small_font.render(
                f"Most similar Greek words from corpus ({len(candidate_words)} candidates):",
                True,
                BLACK,
            ),
            (sidebar_x, sidebar_y),
        )
        sidebar_y += 30
        for other, sim in similarities[:10]:
            sim_text = small_font.render(f"{other}: {sim:.2f}", True, BLACK)
            screen.blit(sim_text, (sidebar_x, sidebar_y))
            sidebar_y += 25
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for rect, greek in greek_rects:
                if rect.collidepoint(mouse_pos):
                    selected_greek = greek
                    break
    
    pygame.display.flip()

pygame.quit()