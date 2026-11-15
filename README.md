# Music Mood Classifier

A small educational AI project that classifies a song's mood from tempo and energy features using K-Nearest Neighbors.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python music_mood.py --tempo 128 --energy 0.85
```

## Test

```bash
python -m unittest test_music_mood.py
```

The training examples are synthetic and embedded in the source for a compact, self-contained learning project.
