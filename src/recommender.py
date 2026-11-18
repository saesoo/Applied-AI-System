import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# --- Scoring weights (from the Phase 2 Algorithm Recipe) ---
GENRE_WEIGHT = 25
MOOD_WEIGHT = 20
NUMERIC_WEIGHT = 11  # per numeric feature

# Numeric features scored by distance (all on a 0.0-1.0 scale except tempo_bpm).
NUMERIC_FEATURES = ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]

# tempo_bpm isn't on a 0.0-1.0 scale, so normalize its distance against an
# assumed catalog spread before comparing it point-for-point with the rest.
TEMPO_RANGE = 150.0

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Score a Song against a UserProfile. Returns (score, reasons)."""
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += GENRE_WEIGHT
            reasons.append(f"Genre matches your preferred '{user.favorite_genre}'")

        if song.mood == user.favorite_mood:
            score += MOOD_WEIGHT
            reasons.append(f"Mood matches your preferred '{user.favorite_mood}'")

        energy_points = NUMERIC_WEIGHT * max(0.0, 1.0 - abs(song.energy - user.target_energy))
        if energy_points > 0:
            score += energy_points
            reasons.append(
                f"Energy is close to your target ({song.energy} vs {user.target_energy})"
            )

        # likes_acoustic: reward high acousticness if they like acoustic, low if not.
        acoustic_target = 0.8 if user.likes_acoustic else 0.2
        acoustic_points = NUMERIC_WEIGHT * max(0.0, 1.0 - abs(song.acousticness - acoustic_target))
        if acoustic_points > 0:
            score += acoustic_points
            fit = "acoustic" if user.likes_acoustic else "non-acoustic"
            reasons.append(f"Fits your taste for {fit} tracks")

        return score, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked = sorted(
            self.songs,
            key=lambda song: self._score(user, song)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = self._score(user, song)
        if not reasons:
            return f"'{song.title}' doesn't strongly match your current preferences."
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Only the preference keys present in user_prefs are scored, so partial and
    full taste profiles both work. Returns (score, reasons).
    """
    score = 0.0
    reasons: List[str] = []

    if "genre" in user_prefs and song["genre"] == user_prefs["genre"]:
        score += GENRE_WEIGHT
        reasons.append(f"Genre matches your preferred '{user_prefs['genre']}'")

    if "mood" in user_prefs and song["mood"] == user_prefs["mood"]:
        score += MOOD_WEIGHT
        reasons.append(f"Mood matches your preferred '{user_prefs['mood']}'")

    for feature in NUMERIC_FEATURES:
        if feature not in user_prefs:
            continue
        target = user_prefs[feature]
        value = song[feature]
        # tempo_bpm lives on a different scale, so normalize its distance.
        spread = TEMPO_RANGE if feature == "tempo_bpm" else 1.0
        distance = abs(value - target) / spread
        points = NUMERIC_WEIGHT * max(0.0, 1.0 - distance)
        if points > 0:
            score += points
            label = feature.replace("_", " ").title()
            reasons.append(f"{label} is close to your target ({value} vs {target})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    ranked = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        ranked.append((song, score, "; ".join(reasons)))

    ranked.sort(key=lambda recommendation: recommendation[1], reverse=True)
    return ranked[:k]
