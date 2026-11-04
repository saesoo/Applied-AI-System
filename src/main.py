"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: target values for each feature the recommender compares against.
    # genre/mood are matched for an exact (or close) match; the rest are numeric
    # targets (0.0-1.0, except tempo_bpm) compared via distance to the song's value.
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.80,
        "tempo_bpm": 120,
        "valence": 0.85,
        "danceability": 0.80,
        "acousticness": 0.15,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
