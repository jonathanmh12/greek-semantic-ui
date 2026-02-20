import re

import numpy as np
import pandas as pd
import pygame
import torch
from sentence_transformers import SentenceTransformer, models

model_dir = "models/GreekBERT_v3"
dataset_url = (
    "https://huggingface.co/datasets/hmcgovern/original-language-bibles-greek/"
    "resolve/main/data/train-00000-of-00001.parquet"
)


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


def load_candidate_words_from_dataframe(df: pd.DataFrame) -> list[str]:
    candidates: set[str] = set()
    for text in df["text"].dropna().astype(str):
        for token in extract_words(text):
            if is_greek_word(token):
                candidates.add(token)
    return sorted(candidates)


def build_embedding_index(texts: list[str]) -> dict[str, np.ndarray]:
    if not texts:
        return {}
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


def parse_reference(reference: str) -> tuple[str, int, int, int] | None:
    parts = reference.split(".")
    if len(parts) < 4:
        return None
    book = parts[0]
    try:
        chapter = int(parts[1])
        verse = int(parts[2])
        token_order = int(parts[3])
    except ValueError:
        return None
    return book, chapter, verse, token_order


def build_verse_lookup(df: pd.DataFrame) -> tuple[
    dict[str, dict[int, list[int]]],
    dict[tuple[str, int, int], list[dict[str, str]]],
]:
    verse_options: dict[str, dict[int, set[int]]] = {}
    verse_tokens: dict[tuple[str, int, int], list[tuple[int, dict[str, str]]]] = {}

    for row in df[["reference", "text", "translation"]].itertuples(index=False):
        parsed = parse_reference(str(row.reference))
        if parsed is None:
            continue
        book, chapter, verse, token_order = parsed

        verse_options.setdefault(book, {}).setdefault(chapter, set()).add(verse)
        verse_tokens.setdefault((book, chapter, verse), []).append(
            (
                token_order,
                {
                    "greek": str(row.text),
                    "english": str(row.translation),
                },
            )
        )

    normalized_options: dict[str, dict[int, list[int]]] = {}
    for book, chapter_map in verse_options.items():
        normalized_options[book] = {
            chapter: sorted(list(verse_set)) for chapter, verse_set in chapter_map.items()
        }

    normalized_tokens: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for key, token_items in verse_tokens.items():
        ordered = sorted(token_items, key=lambda item: item[0])
        normalized_tokens[key] = [token for _, token in ordered]

    return normalized_options, normalized_tokens


class Dropdown:
    def __init__(self, x: int, y: int, width: int, height: int, label: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.option_height = height
        self.label = label
        self.options: list[str] = []
        self.selected_index = 0
        self.open = False
        self.max_visible = 8
        self.scroll_index = 0

    def set_options(self, options: list[str], selected_value: str | None = None) -> None:
        self.options = options
        self.scroll_index = 0
        self.open = False

        if not self.options:
            self.selected_index = 0
            return

        if selected_value is not None and selected_value in self.options:
            self.selected_index = self.options.index(selected_value)
        else:
            self.selected_index = 0

        if self.selected_index >= self.max_visible:
            self.scroll_index = self.selected_index - self.max_visible + 1

    def selected(self) -> str | None:
        if not self.options:
            return None
        return self.options[self.selected_index]

    def list_rect(self) -> pygame.Rect:
        visible_count = min(self.max_visible, len(self.options))
        return pygame.Rect(
            self.rect.x,
            self.rect.y + self.option_height,
            self.rect.width,
            visible_count * self.option_height,
        )

    def draw(self, surface, font, small_font, fg, bg, border) -> None:
        label_text = small_font.render(self.label, True, fg)
        surface.blit(label_text, (self.rect.x, self.rect.y - 18))

        pygame.draw.rect(surface, bg, self.rect)
        pygame.draw.rect(surface, border, self.rect, 2)

        current_value = self.selected() or "-"
        rendered = font.render(current_value, True, fg)
        surface.blit(rendered, (self.rect.x + 8, self.rect.y + 6))

        pygame.draw.polygon(
            surface,
            fg,
            [
                (self.rect.right - 18, self.rect.y + 12),
                (self.rect.right - 8, self.rect.y + 12),
                (self.rect.right - 13, self.rect.y + 20),
            ],
        )

        if self.open and self.options:
            visible_options = self.options[
                self.scroll_index : self.scroll_index + self.max_visible
            ]
            for i, option in enumerate(visible_options):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i + 1) * self.option_height,
                    self.rect.width,
                    self.option_height,
                )
                pygame.draw.rect(surface, bg, option_rect)
                pygame.draw.rect(surface, border, option_rect, 1)

                absolute_index = self.scroll_index + i
                if absolute_index == self.selected_index:
                    pygame.draw.rect(surface, (220, 230, 255), option_rect)

                option_text = font.render(option, True, fg)
                surface.blit(option_text, (option_rect.x + 8, option_rect.y + 6))

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.open = not self.open
            return True

        if self.open and self.options:
            visible_options = self.options[
                self.scroll_index : self.scroll_index + self.max_visible
            ]
            for i in range(len(visible_options)):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i + 1) * self.option_height,
                    self.rect.width,
                    self.option_height,
                )
                if option_rect.collidepoint(pos):
                    self.selected_index = self.scroll_index + i
                    self.open = False
                    return True

        if self.open:
            self.open = False
        return False

    def handle_wheel(self, y_delta: int, mouse_pos: tuple[int, int]) -> bool:
        if not self.open or len(self.options) <= self.max_visible:
            return False

        if not (self.rect.collidepoint(mouse_pos) or self.list_rect().collidepoint(mouse_pos)):
            return False

        if y_delta > 0:
            self.scroll_index = max(0, self.scroll_index - 1)
        elif y_delta < 0:
            max_scroll = len(self.options) - self.max_visible
            self.scroll_index = min(max_scroll, self.scroll_index + 1)
        return True


# Load the underlying transformer
transformer = models.Transformer(
    model_name_or_path=model_dir,
    tokenizer_args={"use_fast": True},
    model_args={"trust_remote_code": False},
)

pooling_model = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode="mean",
)

model = SentenceTransformer(modules=[transformer, pooling_model])

# Optional: move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataframe and build verse lookup once
df = pd.read_parquet(dataset_url)
verse_option_map, verse_token_map = build_verse_lookup(df)

book_options = sorted(list(verse_option_map.keys()))
selected_book = book_options[0]
chapter_options = sorted(list(verse_option_map[selected_book].keys()))
selected_chapter = chapter_options[0]
verse_options = verse_option_map[selected_book][selected_chapter]
selected_verse = verse_options[0]

verse_data = verse_token_map.get((selected_book, selected_chapter, selected_verse), [])

# Build corpus embeddings once from dataframe
candidate_words = load_candidate_words_from_dataframe(df)
candidate_embeddings = build_embedding_index(candidate_words)

# Display verse embeddings (rebuilt only on Display click)
display_embeddings = build_embedding_index([item["greek"] for item in verse_data])


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


pygame.init()
screen = pygame.display.set_mode((1200, 700))
pygame.display.set_caption("Greek Verse UI with Embeddings")
font = pygame.font.SysFont("arial", 20)
small_font = pygame.font.SysFont("arial", 16)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
GREEN = (30, 140, 30)

start_x = 50
start_y = 160
item_width = 120
separator_width = 10
scroll_step = 60

book_dropdown = Dropdown(50, 35, 180, 30, "Book")
chapter_dropdown = Dropdown(250, 35, 130, 30, "Chapter")
verse_dropdown = Dropdown(400, 35, 130, 30, "Verse")
display_button = pygame.Rect(550, 35, 110, 30)

book_dropdown.set_options(book_options, str(selected_book))
chapter_dropdown.set_options([str(c) for c in chapter_options], str(selected_chapter))
verse_dropdown.set_options([str(v) for v in verse_options], str(selected_verse))

greek_rects = []
selected_greek = None
similarities = []
verse_scroll_x = 0


def get_max_verse_scroll() -> int:
    viewport_width = screen.get_width() - (2 * start_x)
    content_width = len(verse_data) * item_width
    return max(0, content_width - viewport_width)


def clamp_verse_scroll() -> None:
    global verse_scroll_x
    verse_scroll_x = max(0, min(verse_scroll_x, get_max_verse_scroll()))


def refresh_chapter_and_verse_options() -> None:
    global selected_chapter, selected_verse

    chapters = sorted(list(verse_option_map[selected_book].keys()))
    if selected_chapter not in chapters:
        selected_chapter = chapters[0]
    chapter_dropdown.set_options([str(c) for c in chapters], str(selected_chapter))

    verses = verse_option_map[selected_book][selected_chapter]
    if selected_verse not in verses:
        selected_verse = verses[0]
    verse_dropdown.set_options([str(v) for v in verses], str(selected_verse))


def refresh_verse_display_data() -> None:
    global verse_data, display_embeddings, selected_greek, verse_scroll_x
    verse_data = verse_token_map.get((selected_book, selected_chapter, selected_verse), [])
    if verse_data:
        display_embeddings = build_embedding_index([item["greek"] for item in verse_data])
    else:
        display_embeddings = {}
    selected_greek = None
    verse_scroll_x = 0


running = True
while running:
    screen.fill(WHITE)
    clamp_verse_scroll()

    book_dropdown.draw(screen, small_font, small_font, BLACK, LIGHT_GRAY, GRAY)
    chapter_dropdown.draw(screen, small_font, small_font, BLACK, LIGHT_GRAY, GRAY)
    verse_dropdown.draw(screen, small_font, small_font, BLACK, LIGHT_GRAY, GRAY)

    pygame.draw.rect(screen, GREEN, display_button)
    pygame.draw.rect(screen, BLACK, display_button, 2)
    display_text = small_font.render("Display", True, WHITE)
    screen.blit(display_text, (display_button.x + 24, display_button.y + 7))

    selection_text = small_font.render(
        f"Current selection: {selected_book} {selected_chapter}:{selected_verse}",
        True,
        BLACK,
    )
    screen.blit(selection_text, (50, 80))

    scroll_help = small_font.render(
        "Scroll verse: mouse wheel over verse area or Left/Right arrow keys",
        True,
        BLACK,
    )
    screen.blit(scroll_help, (50, 104))

    viewport_rect = pygame.Rect(start_x, start_y - 12, screen.get_width() - (2 * start_x), 84)
    pygame.draw.rect(screen, LIGHT_GRAY, viewport_rect)
    pygame.draw.rect(screen, GRAY, viewport_rect, 1)

    greek_rects = []
    x = start_x - verse_scroll_x
    screen.set_clip(viewport_rect)
    for item in verse_data:
        greek_text = font.render(item["greek"], True, BLUE)
        rect = greek_text.get_rect(topleft=(x, start_y))
        screen.blit(greek_text, rect)

        english_text = small_font.render(item["english"], True, BLACK)
        screen.blit(english_text, (x, start_y + 32))

        if rect.colliderect(viewport_rect):
            greek_rects.append((rect, item["greek"]))

        if x > start_x:
            pygame.draw.line(
                screen,
                GRAY,
                (x - separator_width // 2, start_y - 8),
                (x - separator_width // 2, start_y + 55),
                2,
            )

        x += item_width
    screen.set_clip(None)

    max_scroll = get_max_verse_scroll()
    if max_scroll > 0:
        track_rect = pygame.Rect(start_x, start_y + 78, viewport_rect.width, 8)
        thumb_width = max(40, int(track_rect.width * (viewport_rect.width / (len(verse_data) * item_width))))
        thumb_x = track_rect.x + int((verse_scroll_x / max_scroll) * (track_rect.width - thumb_width))
        thumb_rect = pygame.Rect(thumb_x, track_rect.y, thumb_width, track_rect.height)
        pygame.draw.rect(screen, GRAY, track_rect)
        pygame.draw.rect(screen, BLUE, thumb_rect)

    if selected_greek:
        similarities = []
        selected_emb = display_embeddings.get(selected_greek, get_embedding(selected_greek))

        for other_greek, other_emb in candidate_embeddings.items():
            if other_greek != selected_greek:
                sim = cosine_similarity(selected_emb, other_emb)
                similarities.append((other_greek, sim))

        similarities.sort(key=lambda item: item[1], reverse=True)

        sidebar_x = 50
        sidebar_y = 280
        screen.blit(
            font.render(f"Selected: {selected_greek}", True, BLACK),
            (sidebar_x, sidebar_y),
        )
        sidebar_y += 36

        screen.blit(
            small_font.render(
                f"Most similar Greek words from dataframe corpus ({len(candidate_words)} candidates):",
                True,
                BLACK,
            ),
            (sidebar_x, sidebar_y),
        )
        sidebar_y += 26

        for other, sim in similarities[:10]:
            sim_text = small_font.render(f"{other}: {sim:.2f}", True, BLACK)
            screen.blit(sim_text, (sidebar_x, sidebar_y))
            sidebar_y += 22

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            if book_dropdown.handle_wheel(event.y, mouse_pos):
                continue
            if chapter_dropdown.handle_wheel(event.y, mouse_pos):
                continue
            if verse_dropdown.handle_wheel(event.y, mouse_pos):
                continue

            viewport_rect = pygame.Rect(start_x, start_y - 12, screen.get_width() - (2 * start_x), 84)
            if viewport_rect.collidepoint(mouse_pos):
                verse_scroll_x -= event.y * scroll_step
                clamp_verse_scroll()
                continue

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                verse_scroll_x += scroll_step
                clamp_verse_scroll()
                continue
            if event.key == pygame.K_LEFT:
                verse_scroll_x -= scroll_step
                clamp_verse_scroll()
                continue
            if event.key == pygame.K_HOME:
                verse_scroll_x = 0
                continue
            if event.key == pygame.K_END:
                verse_scroll_x = get_max_verse_scroll()
                continue

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            if book_dropdown.handle_click(mouse_pos):
                current = book_dropdown.selected()
                if current is not None:
                    selected_book = current
                    refresh_chapter_and_verse_options()
                chapter_dropdown.open = False
                verse_dropdown.open = False
                continue

            if chapter_dropdown.handle_click(mouse_pos):
                current = chapter_dropdown.selected()
                if current is not None:
                    selected_chapter = int(current)
                    refresh_chapter_and_verse_options()
                book_dropdown.open = False
                verse_dropdown.open = False
                continue

            if verse_dropdown.handle_click(mouse_pos):
                current = verse_dropdown.selected()
                if current is not None:
                    selected_verse = int(current)
                book_dropdown.open = False
                chapter_dropdown.open = False
                continue

            if display_button.collidepoint(mouse_pos):
                refresh_verse_display_data()
                book_dropdown.open = False
                chapter_dropdown.open = False
                verse_dropdown.open = False
                continue

            for rect, greek in greek_rects:
                if rect.collidepoint(mouse_pos):
                    selected_greek = greek
                    break

            book_dropdown.open = False
            chapter_dropdown.open = False
            verse_dropdown.open = False

pygame.quit()
