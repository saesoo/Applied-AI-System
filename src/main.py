"""
Command line runner for the Music Recommender Simulation.

The app is driven by a Retrieval-Augmented Generation (RAG) loop: you type a
natural-language request, the system retrieves matching songs from the catalog
(data/songs.csv) and an AI assistant answers grounded in those retrieved songs.

See src/rag.py for the retrieval + generation pipeline.
"""

from src.recommender import load_songs
from src.rag import MusicRAG

# Load GEMINI_API_KEY (and anything else) from a local .env file if present.
# .env is gitignored, so the key never gets committed. No-op if python-dotenv
# isn't installed or there's no .env file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    songs = load_songs("data/songs.csv")
    if not songs:
        print("No songs loaded. Check data/songs.csv.")
        return

    # Build the retrieval corpus from multiple sources: the song catalog plus
    # any custom reference documents in data/notes/.
    rag = MusicRAG(songs, notes_dir="data/notes")

    print("\n🎵 Music Recommender (RAG mode)")
    print("Sources indexed: " + ", ".join(
        f"{src} ({n})" for src, n in rag.sources.items()
    ))
    if rag.generation_available:
        print("AI generation: ON (grounded in the song catalog)")
    else:
        print(
            "AI generation: OFF - set GEMINI_API_KEY to enable grounded "
            "AI recommendations. Running in retrieval-only mode."
        )
    print("Describe what you want to hear, or type 'quit' to exit.")
    print("Example: 'upbeat pop to get hyped for a workout'\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\n" + rag.answer(query) + "\n")


if __name__ == "__main__":
    main()
