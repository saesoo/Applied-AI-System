"""
Tests for the RAG retrieval pipeline.

These run fully offline (no API key / network) by constructing MusicRAG with
client=None, which forces retrieval-only mode. They cover the deterministic
retrieval + grounding behavior across multiple data sources, not the LLM
generation itself.
"""

from src.rag import Document, MusicRAG, documents_from_text_dir


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


def make_rag(**kwargs):
    # client=None forces offline, retrieval-only mode (no API calls).
    return MusicRAG(make_songs(), client=None, **kwargs)


def test_offline_mode_when_no_client():
    rag = make_rag()
    assert rag.generation_available is False


def test_songs_are_loaded_as_documents():
    rag = make_rag()
    assert rag.sources["songs"] == 3


def test_retrieve_ranks_matching_genre_and_mood_first():
    rag = make_rag()
    results = rag.retrieve("chill lofi for studying", k=2)
    assert results, "expected at least one retrieved document"
    top_doc, top_score = results[0]
    assert top_doc.source == "songs"
    assert top_doc.payload["genre"] == "lofi"
    assert top_score > 0


def test_intent_word_expands_to_catalog_labels():
    # "workout" isn't a genre or mood, but should pull the high-energy track
    # via the intent-hint expansion.
    rag = make_rag()
    titles = [doc.title for doc, _ in rag.retrieve("music for a workout", k=3)]
    assert "Storm Runner" in titles


def test_retrieve_respects_k():
    rag = make_rag()
    assert len(rag.retrieve("happy pop", k=1)) <= 1


def test_answer_offline_is_grounded_in_retrieved_documents():
    rag = make_rag()
    answer = rag.answer("chill lofi for studying")
    assert isinstance(answer, str) and answer.strip() != ""
    # Offline answer must only mention real catalog songs, never invented ones.
    assert "Midnight Coding" in answer


# --- multiple data sources ------------------------------------------------

def test_retrieves_across_multiple_sources_from_documents_arg():
    note = Document(
        doc_id="guide-1",
        source="guide",
        title="Workout Guide",
        text="high energy intense pop tracks for a workout and gym sessions",
        payload={"text": "..."},
    )
    rag = MusicRAG(make_songs(), documents=[note], client=None)
    assert rag.sources.get("guide") == 1

    results = rag.retrieve("workout", k=5)
    retrieved_sources = {doc.source for doc, _ in results}
    # Retrieval spans both the songs catalog and the custom document.
    assert "guide" in retrieved_sources
    assert "songs" in retrieved_sources


def test_loads_custom_text_documents_from_directory(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "study_guide.md").write_text(
        "chill lofi beats that are perfect for studying and deep focus"
    )

    docs = documents_from_text_dir(str(notes))
    assert len(docs) == 1
    assert docs[0].source == "notes"

    rag = MusicRAG(make_songs(), notes_dir=str(notes), client=None)
    assert rag.sources.get("notes") == 1
    sources = {doc.source for doc, _ in rag.retrieve("studying", k=5)}
    assert "notes" in sources


def test_missing_notes_dir_is_safe():
    rag = make_rag(notes_dir="data/this_dir_does_not_exist")
    assert "notes" not in rag.sources
    assert rag.sources["songs"] == 3
