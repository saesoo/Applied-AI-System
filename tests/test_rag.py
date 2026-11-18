"""
Tests for the RAG retrieval pipeline.

These run fully offline (no API key / network) by constructing MusicRAG with
client=None, which forces retrieval-only mode. They cover the deterministic
retrieval + grounding behavior, not the LLM generation itself.
"""

from src.rag import MusicRAG


def make_songs():
    return [
        {
            "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
            "genre": "pop", "mood": "happy", "energy": 0.82, "tempo_bpm": 118,
            "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
        },
        {
            "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
            "genre": "lofi", "mood": "chill", "energy": 0.42, "tempo_bpm": 78,
            "valence": 0.56, "danceability": 0.62, "acousticness": 0.71,
        },
        {
            "id": 3, "title": "Storm Runner", "artist": "Voltline",
            "genre": "rock", "mood": "intense", "energy": 0.91, "tempo_bpm": 152,
            "valence": 0.48, "danceability": 0.66, "acousticness": 0.10,
        },
    ]


def make_rag():
    # client=None forces offline, retrieval-only mode (no API calls).
    return MusicRAG(make_songs(), client=None)


def test_offline_mode_when_no_client():
    rag = make_rag()
    assert rag.generation_available is False


def test_retrieve_ranks_matching_genre_and_mood_first():
    rag = make_rag()
    results = rag.retrieve("chill lofi for studying", k=2)
    assert results, "expected at least one retrieved song"
    top_song, top_score = results[0]
    assert top_song["genre"] == "lofi"
    assert top_score > 0


def test_intent_word_expands_to_catalog_labels():
    # "workout" isn't a genre or mood, but should pull the high-energy track
    # via the intent-hint expansion.
    rag = make_rag()
    titles = [song["title"] for song, _ in rag.retrieve("music for a workout", k=3)]
    assert "Storm Runner" in titles


def test_retrieve_respects_k():
    rag = make_rag()
    assert len(rag.retrieve("happy pop", k=1)) <= 1


def test_answer_offline_is_grounded_in_retrieved_songs():
    rag = make_rag()
    answer = rag.answer("chill lofi for studying")
    assert isinstance(answer, str) and answer.strip() != ""
    # Offline answer must only mention real catalog songs, never invented ones.
    assert "Midnight Coding" in answer
