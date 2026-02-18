import pygame
import numpy as np
import torch
import os

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

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer, models

model_dir = "models/Greek_v2_Expanded"

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

# Now this works beautifully:
def get_embedding(text: str):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    # normalize_embeddings=True is often helpful for cosine similarity

# Cache as before
embeddings = {item["greek"]: get_embedding(item["greek"]) for item in verse_data}

# Function to get embedding
def get_embedding(text):
    with torch.no_grad():
        return model.encode(text)

# Compute embeddings for all Greek phrases (cache them)
embeddings = {item["greek"]: get_embedding(item["greek"]) for item in verse_data}

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
        selected_emb = embeddings[selected_greek]
        for other_greek in embeddings:
            if other_greek != selected_greek:
                sim = cosine_similarity(selected_emb, embeddings[other_greek])
                similarities.append((other_greek, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        sidebar_x = 50
        sidebar_y = 200
        screen.blit(font.render(f"Selected: {selected_greek}", True, BLACK), (sidebar_x, sidebar_y))
        sidebar_y += 40
        screen.blit(small_font.render("Most similar Greek phrases (cosine):", True, BLACK), (sidebar_x, sidebar_y))
        sidebar_y += 30
        for other, sim in similarities[:5]:
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