"""
Retrieval-Augmented Generation (RAG) for the Music Recommender.

This turns the recommender from "score a fixed preferences dict -> print a list"
into a system that answers *natural-language* requests grounded in the song
catalog:

    1. RETRIEVE  - treat data/songs.csv as a knowledge base and pull the songs
                   most relevant to the user's free-text request.
    2. AUGMENT   - inject ONLY those retrieved songs into the prompt as context.
    3. GENERATE  - Google Gemini writes a recommendation that actively uses the
                   retrieved song data, and is instructed to recommend nothing
                   that wasn't retrieved (grounding -> no hallucinated songs).

If no Gemini API key is configured, the generation step degrades gracefully
to a retrieval-only response so the app still runs offline.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

MODEL = "gemini-2.5-flash"

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


class MusicRAG:
    """RAG pipeline over a list of song dicts (as produced by load_songs)."""

    def __init__(
        self,
        songs: List[Dict],
        client: Optional[object] = None,
        model: str = MODEL,
    ):
        self.songs = songs
        self.model = model
        # Precompute a searchable "document" per song for the retrieval step.
        self._docs = [self._song_document(song) for song in songs]
        # Resolve a client lazily; None means run in offline (retrieval-only) mode.
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

    # --- retrieval --------------------------------------------------------

    @staticmethod
    def _song_document(song: Dict) -> List[str]:
        """The text we match a query against: the song's categorical metadata."""
        return _tokenize(
            f"{song['title']} {song['artist']} {song['genre']} {song['mood']}"
        )

    def _expand_query(self, query: str) -> List[str]:
        """Query terms plus any catalog labels implied by intent words."""
        terms = _tokenize(query)
        expanded = list(terms)
        for term in terms:
            expanded.extend(_INTENT_HINTS.get(term, []))
        return expanded

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Dict, float]]:
        """
        Return the top-k (song, relevance) pairs for a natural-language query.

        Relevance is lexical overlap between the expanded query and each song's
        genre/mood/title/artist, with genre and mood matches weighted higher
        since they're the strongest taste signal in this catalog.
        """
        terms = self._expand_query(query)
        ranked: List[Tuple[Dict, float]] = []
        for song, doc in zip(self.songs, self._docs):
            score = 0.0
            for term in terms:
                if term == song["genre"].lower():
                    score += 3.0
                elif term == song["mood"].lower():
                    score += 2.5
                elif term in doc:
                    score += 1.0
            ranked.append((song, score))

        ranked.sort(key=lambda pair: pair[1], reverse=True)

        # If nothing matched (e.g. a vague query), fall back to the full catalog
        # order so the generator still has candidates to work with.
        if ranked and ranked[0][1] == 0.0:
            return ranked[:k]
        return [pair for pair in ranked if pair[1] > 0][:k]

    # --- augmentation -----------------------------------------------------

    @staticmethod
    def _format_context(retrieved: List[Tuple[Dict, float]]) -> str:
        """Render retrieved songs as the grounding context block for the prompt."""
        lines = []
        for song, _ in retrieved:
            lines.append(
                f"- \"{song['title']}\" by {song['artist']} "
                f"(genre: {song['genre']}, mood: {song['mood']}, "
                f"energy: {song['energy']}, tempo: {song['tempo_bpm']} bpm, "
                f"valence: {song['valence']}, danceability: {song['danceability']}, "
                f"acousticness: {song['acousticness']})"
            )
        return "\n".join(lines)

    # --- generation -------------------------------------------------------

    _SYSTEM_PROMPT = (
        "You are a music recommendation assistant for a small catalog. "
        "You will be given a user's request and a list of CANDIDATE SONGS "
        "retrieved from the catalog. Recommend ONLY songs from that candidate "
        "list - never invent songs, artists, or details that are not shown. "
        "Pick the best matches for the request, order them best-first, and for "
        "each one give a short reason that references the song's actual "
        "attributes (genre, mood, energy, tempo, etc.). If none of the "
        "candidates fit the request well, say so honestly."
    )

    def answer(self, query: str, k: int = 4) -> str:
        """
        Full RAG answer to a natural-language query.

        Retrieves relevant songs, augments the prompt with them, and asks Claude
        to generate a grounded recommendation. Falls back to a retrieval-only
        response when generation isn't available.
        """
        retrieved = self.retrieve(query, k=k)
        context = self._format_context(retrieved)

        if not self.generation_available:
            return self._offline_answer(retrieved)

        user_prompt = (
            f"User request: {query}\n\n"
            f"CANDIDATE SONGS (the only songs you may recommend):\n{context}\n\n"
            "Recommend the best matches for the request."
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
    def _offline_answer(retrieved: List[Tuple[Dict, float]]) -> str:
        """Retrieval-only response used when AI generation is unavailable."""
        if not retrieved:
            return "No songs in the catalog matched your request."
        lines = ["Retrieved from the catalog (AI generation unavailable):"]
        for song, score in retrieved:
            lines.append(
                f"  • {song['title']} by {song['artist']} "
                f"[{song['genre']}/{song['mood']}] (relevance {score:.1f})"
            )
        return "\n".join(lines)
