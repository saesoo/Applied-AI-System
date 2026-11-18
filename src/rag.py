"""
Retrieval-Augmented Generation (RAG) for the Music Recommender.

This turns the recommender from "score a fixed preferences dict -> print a list"
into a system that answers *natural-language* requests grounded in a corpus
built from MULTIPLE data sources:

    * the song catalog (data/songs.csv) - one document per song
    * optional custom text documents (data/notes/*.md|*.txt) - guides, artist
      bios, playlist notes, or any reference material you drop in
    * any extra Document objects you pass in programmatically

Pipeline:

    1. RETRIEVE  - rank every document across all sources by relevance to the
                   user's free-text request.
    2. AUGMENT   - inject ONLY the retrieved documents into the prompt as
                   context, tagged by source.
    3. GENERATE  - Google Gemini writes a recommendation that actively uses the
                   retrieved data, recommends only songs that appear in the
                   context, and may use the notes as supporting background.

If no Gemini API key is configured, the generation step degrades gracefully
to a retrieval-only response so the app still runs offline.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MODEL = "gemini-3.8-flash"

# Words that carry no retrieval signal.
_STOPWORDS = {
    "a", "an", "the", "some", "any", "me", "i", "for", "to", "of", "and", "or",
    "with", "that", "this", "my", "want", "need", "like", "give", "find",
    "something", "songs", "song", "music", "track", "tracks", "please", "can",
    "you", "recommend", "recommendation", "play", "listen", "mood", "feel",
    "feeling", "is", "are", "in", "on", "it",
}

# Maps common natural-language intent words to the catalog's categorical labels.
# Lets a query like "upbeat workout music" retrieve high-energy songs even when
# the exact genre/mood words aren't typed.
_INTENT_HINTS: Dict[str, List[str]] = {
    "upbeat": ["happy", "pop", "intense"],
    "energetic": ["intense", "happy"],
    "workout": ["intense", "happy"],
    "gym": ["intense", "happy"],
    "hype": ["intense"],
    "pump": ["intense"],
    "chill": ["chill", "lofi"],
    "relax": ["chill", "lofi"],
    "relaxing": ["chill", "lofi"],
    "study": ["chill", "lofi"],
    "studying": ["chill", "lofi"],
    "focus": ["chill", "lofi"],
    "calm": ["chill", "lofi"],
    "sad": ["chill"],
    "happy": ["happy"],
    "sleep": ["chill", "lofi"],
}


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


@dataclass
class Document:
    """
    A single retrievable unit from any data source.

    `source` labels where it came from ("songs", "notes", or a custom tag),
    `text` is what retrieval matches against, and `payload` holds the original
    structured data (the song dict for songs, or the raw text for notes).
    """
    doc_id: str
    source: str
    title: str
    text: str
    payload: Dict = field(default_factory=dict)


# --- data-source loaders --------------------------------------------------

def documents_from_songs(songs: List[Dict]) -> List[Document]:
    """Turn the song catalog (list of dicts) into retrievable documents."""
    docs = []
    for song in songs:
        text = f"{song['title']} {song['artist']} {song['genre']} {song['mood']}"
        docs.append(
            Document(
                doc_id=f"song-{song['id']}",
                source="songs",
                title=song["title"],
                text=text,
                payload=song,
            )
        )
    return docs


def documents_from_text_dir(path: str, source: str = "notes") -> List[Document]:
    """
    Load every .md / .txt file in a directory as a custom reference document.

    Each file becomes one document whose full text is searchable. Missing
    directories are treated as "no documents" so the app still runs.
    """
    docs = []
    if not path or not os.path.isdir(path):
        return docs
    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        full = os.path.join(path, fname)
        with open(full, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        docs.append(
            Document(
                doc_id=f"{source}-{fname}",
                source=source,
                title=fname.rsplit(".", 1)[0].replace("_", " ").title(),
                text=content,
                payload={"text": content, "filename": fname},
            )
        )
    return docs


class MusicRAG:
    """RAG pipeline over a multi-source document corpus."""

    def __init__(
        self,
        songs: List[Dict],
        notes_dir: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        client: Optional[object] = None,
        model: str = MODEL,
    ):
        # Build the corpus from every configured source.
        self.corpus: List[Document] = documents_from_songs(songs)
        self.corpus += documents_from_text_dir(notes_dir) if notes_dir else []
        self.corpus += documents or []

        # Precompute a token set per document for fast relevance scoring.
        self._doc_tokens = {
            doc.doc_id: set(_tokenize(doc.text)) for doc in self.corpus
        }

        self.model = model
        self.client = client if client is not None else self._make_client()

    # --- client setup -----------------------------------------------------

    @staticmethod
    def _make_client() -> Optional[object]:
        """Return a Gemini client, or None if unavailable (offline mode)."""
        try:
            from google import genai
        except ImportError:
            return None
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            return None

    @property
    def generation_available(self) -> bool:
        return self.client is not None

    @property
    def sources(self) -> Dict[str, int]:
        """Document count per source, e.g. {'songs': 10, 'notes': 3}."""
        counts: Dict[str, int] = {}
        for doc in self.corpus:
            counts[doc.source] = counts.get(doc.source, 0) + 1
        return counts

    # --- retrieval --------------------------------------------------------

    def _expand_query(self, query: str) -> List[str]:
        """Query terms plus any catalog labels implied by intent words."""
        terms = _tokenize(query)
        expanded = list(terms)
        for term in terms:
            expanded.extend(_INTENT_HINTS.get(term, []))
        return expanded

    def _score_document(self, doc: Document, terms: List[str]) -> float:
        """
        Relevance of one document to the expanded query terms.

        Song documents weight exact genre/mood matches higher (strongest taste
        signal). All sources additionally earn points for term presence in their
        text, so free-text notes are ranked on the same scale.
        """
        tokens = self._doc_tokens[doc.doc_id]
        score = 0.0
        for term in terms:
            if doc.source == "songs":
                if term == doc.payload["genre"].lower():
                    score += 3.0
                    continue
                if term == doc.payload["mood"].lower():
                    score += 2.5
                    continue
            if term in tokens:
                score += 1.0
        return score

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Return the top-k (document, relevance) pairs across ALL sources for a
        natural-language query, best-first.
        """
        terms = self._expand_query(query)
        ranked = [(doc, self._score_document(doc, terms)) for doc in self.corpus]
        ranked.sort(key=lambda pair: pair[1], reverse=True)

        # If nothing matched (vague query), fall back to the top of the corpus
        # so the generator still has candidates to work with.
        if ranked and ranked[0][1] == 0.0:
            return ranked[:k]
        return [pair for pair in ranked if pair[1] > 0][:k]

    # --- augmentation -----------------------------------------------------

    @staticmethod
    def _render_document(doc: Document) -> str:
        """Render one retrieved document as a context line, tagged by source."""
        if doc.source == "songs":
            s = doc.payload
            return (
                f"- [SONG] \"{s['title']}\" by {s['artist']} "
                f"(genre: {s['genre']}, mood: {s['mood']}, "
                f"energy: {s['energy']}, tempo: {s['tempo_bpm']} bpm, "
                f"valence: {s['valence']}, danceability: {s['danceability']}, "
                f"acousticness: {s['acousticness']})"
            )
        excerpt = " ".join(doc.text.split())
        if len(excerpt) > 300:
            excerpt = excerpt[:300].rstrip() + "..."
        return f"- [{doc.source.upper()}] {doc.title}: {excerpt}"

    def _format_context(self, retrieved: List[Tuple[Document, float]]) -> str:
        """Render retrieved documents as the grounding context block."""
        return "\n".join(self._render_document(doc) for doc, _ in retrieved)

    # --- generation -------------------------------------------------------

    _SYSTEM_PROMPT = (
        "You are a music recommendation assistant for a small catalog. "
        "You will be given a user's request and RETRIEVED CONTEXT drawn from "
        "multiple sources: SONG entries from the catalog and optional reference "
        "NOTES (guides, artist info). Recommend ONLY songs that appear as SONG "
        "entries in the context - never invent songs, artists, or details that "
        "are not shown. You may use the NOTES as supporting background to "
        "explain or justify your picks. Order recommendations best-first, and "
        "for each one give a short reason that references the song's actual "
        "attributes (genre, mood, energy, tempo, etc.). If none of the songs "
        "fit the request well, say so honestly."
    )

    def answer(self, query: str, k: int = 4) -> str:
        """
        Full RAG answer to a natural-language query.

        Retrieves relevant documents across all sources, augments the prompt
        with them, and asks Gemini to generate a grounded recommendation. Falls
        back to a retrieval-only response when generation isn't available.
        """
        retrieved = self.retrieve(query, k=k)
        context = self._format_context(retrieved)

        if not self.generation_available:
            return self._offline_answer(retrieved)

        user_prompt = (
            f"User request: {query}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            "Recommend the best songs for the request."
        )

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )
            text = (response.text or "").strip()
        except Exception as exc:  # network/auth/etc. -> degrade, don't crash
            return (
                f"[AI generation failed: {exc}]\n"
                + self._offline_answer(retrieved)
            )

        return text or self._offline_answer(retrieved)

    @staticmethod
    def _offline_answer(retrieved: List[Tuple[Document, float]]) -> str:
        """Retrieval-only response used when AI generation is unavailable."""
        if not retrieved:
            return "Nothing in the corpus matched your request."
        lines = ["Retrieved from the corpus (AI generation unavailable):"]
        for doc, score in retrieved:
            if doc.source == "songs":
                s = doc.payload
                lines.append(
                    f"  • [song] {s['title']} by {s['artist']} "
                    f"[{s['genre']}/{s['mood']}] (relevance {score:.1f})"
                )
            else:
                lines.append(
                    f"  • [{doc.source}] {doc.title} (relevance {score:.1f})"
                )
        return "\n".join(lines)
