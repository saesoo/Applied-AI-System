# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

I asked Claude Code to implement the three stub functions in `recommender.py` (`load_songs`, `score_song`, `recommend_songs`) one at a time, verify each one actually ran against the real data, and then finish writing up `README.md` and this model card.

**Prompts used:**

- "use Python's csv module to read data/songs.csv and return a list of dictionaries"
- "def score_song(...) ... generate points for a genre match, a mood match, and how to calculate a score for numerical features like energy."
- "finish the readme"
- "go ahead and fill in the model card and ai interations"

**What did the agent generate or change?**

- `src/recommender.py`: implemented `load_songs` (using `csv.DictReader`, converting `id` to `int` and the five numeric columns to `float`), `score_song` (a 100-point recipe: 25 pts exact genre match, 20 pts exact mood match, 11 pts each for energy/valence/danceability/acousticness/tempo_bpm scaled by distance from the user's target), and `recommend_songs` (scores every song, joins the per-feature reasons into one explanation string, sorts by score, returns the top *k*).
- `README.md`: filled in the Sample Recommendation Output, Experiments You Tried, Limitations and Risks, and a draft Reflection, all based on actual output from running the implementation.
- `model_card.md`: filled in all nine sections.

**What did you verify or fix manually?**

After the agent's first pass at `score_song`/`recommend_songs`, running `src/main.py` crashed with a `KeyError: 'valence'` — the committed starter `main.py` only sets a partial `user_prefs` (`genre`, `mood`, `energy`), but the scorer assumed every numeric preference key would always be present. I had the agent fix `score_song` to skip any numeric feature missing from `user_prefs` instead of indexing it directly, then re-ran `src/main.py` end-to-end (both with the partial starter profile and a fully-specified profile) to confirm the rankings and explanations looked correct before accepting the change.

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

<!-- e.g., Strategy, Factory, Observer, etc. -->

**How did AI help you brainstorm or implement it?**

<!-- Describe the conversation or suggestions that led to your decision -->

**How does the pattern appear in your final code?**

<!-- Point to the relevant class or method -->
