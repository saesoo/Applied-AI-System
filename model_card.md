# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeMatch 1.0**

---

## 2. Intended Use  

VibeMatch takes a listener's stated taste profile (favorite genre, favorite mood, and optionally target values for energy, tempo, valence, danceability, and acousticness) and ranks a fixed catalog of songs by how well each one fits that profile. It returns the top *k* matches along with a plain-language explanation of why each song was picked.

It assumes the user can describe their taste as a short set of preferences up front — it does not learn from listening history, skips, or likes over time. It is a classroom simulation built to practice designing a scoring/ranking algorithm and reasoning about its behavior, not a production recommender for real listeners.

---

## 3. How the Model Works  

Every song in the catalog is tagged with a genre, a mood, and five numeric traits (energy, tempo, valence, danceability, acousticness). A listener's taste profile is described the same way: a target genre, a target mood, and optionally target values for those same numeric traits.

To score a song, the model hands out points in two ways. First, it checks whether the song's genre and mood exactly match what the listener asked for — genre is worth up to 25 points and mood up to 20 points if they match, 0 if they don't. Second, for each numeric trait the listener gave a target for, the model looks at how far the song's value is from that target: the closer the two numbers are, the more of that trait's 11 points the song earns, and the score falls off smoothly to 0 the farther apart they are. (Tempo is measured in beats per minute instead of a 0–1 scale, so its distance is rescaled first so it doesn't dominate the other traits.) All of these points are added up into one total score, songs are sorted from highest to lowest, and the top *k* are returned — each with a short list of which matches drove its score, stitched into one explanation sentence.

The starter code only had empty stub functions (`load_songs`, `score_song`, `recommend_songs`) with no real logic. The main changes were: writing the CSV loader with `csv.DictReader`; designing the point-based scoring recipe described above, including the genre/mood vs. numeric weighting split and the tempo rescaling; making the scorer tolerant of partial taste profiles (so a listener who only cares about genre, mood, and energy doesn't break anything by omitting the other traits); and writing the ranking step that sorts and truncates to the top *k*.

---

## 4. Data  

The catalog is the 10 songs in `data/songs.csv`. Genres represented: pop (2), lofi (3), rock (1), ambient (1), jazz (1), synthwave (1), and indie pop (1). Moods represented: chill (3), happy (2), intense (2), relaxed (1), moody (1), and focused (1). No songs were added or removed from the provided dataset.

Musical taste is much richer than what's captured here. There's no artist popularity, release era, language, explicit lyric content, or instrumentation data, and the five numeric traits (energy, tempo, valence, danceability, acousticness) are a small slice of what actually makes a song feel a certain way. With only 10 songs and 7 genres, several genres have exactly one representative — the model can't show whether it generalizes within a genre versus just picking the one song available.

---

## 5. Strengths  

The model works best for listeners whose taste maps cleanly onto one of the catalog's existing genre/mood combinations — e.g. a "pop + happy" profile correctly surfaces `Sunrise City` (pop, happy) at the top of the ranking, well ahead of songs that only share a couple of traits. It also handles graceful degradation well: a listener who only specifies genre, mood, and energy still gets a sensible ranking, just with less differentiation than a listener who fills in all seven preference fields, which matches the intuition that more information should produce a sharper match.

The explanation strings are a real strength — because the score is built from discrete, named components (genre match, mood match, each numeric trait), it's cheap to report *why* a song ranked where it did, in contrast to a black-box score that would need a separate explanation system bolted on.

---

## 6. Limitations and Bias 

The model only considers the seven features in the CSV — nothing about lyrics, instrumentation, artist, era, or language factors into the score, so two songs that a human would consider totally different "vibes" could score identically if their labeled traits happen to match.

Genre and mood matching is exact-string, which quietly disadvantages related-but-differently-labeled songs: `Rooftop Lights` is tagged `indie pop`, so a listener asking for `pop` gets zero genre credit for it even though a human would probably consider it a reasonable match. With genres like `jazz`, `rock`, and `ambient` each represented by exactly one song in the catalog, any listener whose taste centers on those genres is working with a single candidate, not a real choice.

Because genre (25 pts) and mood (20 pts) are worth more than any individual numeric trait (11 pts), the system is biased toward listeners who can state a clean genre/mood preference and against listeners whose taste is better described numerically (e.g. "I like high-energy, upbeat songs regardless of genre") — those listeners can't earn as much total score even for a near-perfect numeric match, because 45 of the 100 possible points are gated behind exact genre/mood labels. This weighting was my own design choice, not something derived from data, so it encodes my assumptions about what matters most in a taste match.

---

## 7. Evaluation  

I ran the recommender against two profiles: the starter profile (`genre=pop, mood=happy, energy=0.80`, nothing else specified) and a fully-specified profile (adding `tempo_bpm=120, valence=0.85, danceability=0.80, acousticness=0.15`). For each, I checked whether the top result made intuitive sense and whether the explanation text matched the actual score breakdown.

Both runs put `Sunrise City` (pop, happy, closely matching numeric traits) at #1, which matched my intuition. What I looked for beyond the top result was whether the *ordering* of the rest of the list made sense — in the full-profile run, `Gym Hero` (pop genre, but wrong mood and tempo) still outranked songs with better numeric alignment but no genre match, which confirmed the genre/mood weighting was working as designed rather than being accidentally drowned out by the numeric terms.

The most interesting finding was how much the ranking changed between the partial and full profile for the same catalog — same top pick, but the score gap between #1 and #2 roughly doubled (55.78 → 34.57 vs. 99.08 → 74.83) once more preferences were supplied, which makes sense since more shared traits means more opportunities to differentiate, but wasn't something I'd have predicted before running it.

---

## 8. Future Work  

- Add a looser genre/mood matching step (e.g. a small hand-built list of related genres, or simple substring matching) so `indie pop` can partially credit a `pop` preference instead of scoring zero.
- Let the explanation also mention *mismatches*, not just matches, so a listener understands what kept a song from scoring higher, not only what helped it.
- Add a diversity constraint to the ranking step (e.g. cap how many songs from the same artist or genre can appear in the top *k*) so the list doesn't end up dominated by one artist that happens to score well on every trait.
- Support richer preferences, like a list of acceptable genres or a preferred range instead of a single target value per numeric trait, so "anything from 100–140 bpm" doesn't have to be approximated by distance-to-120.
- Learn or tune the category/numeric weighting from feedback (e.g. "I liked this, not that") instead of hand-picking the 25/20/11 point split.

---

## 9. Personal Reflection  

Working through this made the mechanics of a recommender feel much less mysterious: it's not magic, it's a scoring function and a sort. Once I had to actually choose how many points genre vs. mood vs. each numeric trait should be worth, I realized that every "ranking algorithm" is really just a stack of small, somewhat arbitrary design decisions, and the system's behavior — and who it serves well — follows directly from those choices.

The most unexpected thing was how much a single labeling decision (`indie pop` vs. `pop`) could silently change an outcome. It's a tiny catalog, so a mismatch like that is easy to spot, but it made very concrete how a real-world system with millions of songs and inconsistent tagging could quietly bury good matches for reasons that have nothing to do with actual musical fit. It changed how I think about recommendation apps: the "taste" they seem to understand is really just however their engineers decided to represent and weight a handful of signals.
