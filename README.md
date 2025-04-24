# Netflix Semantic Recommender -The Maze 

> **Multimodal search & recommender demo for the Netflix catalogue – powered by Sentence-Transformers, BLIP and Gradio.**

Gradio app link [(https://huggingface.co/spaces/karouswissem/the_maze_RS)]

---

## ✨ Features

| Input mode | Pipeline                                    | What happens                                                                 |
|------------|---------------------------------------------|------------------------------------------------------------------------------|
| **Text**   | `SentenceTransformer(all-MiniLM-L6-v2)`     | Your query → 384-d embedding → cosine-similarity against precalculated title/overview vectors |
| **Image**  | `BLIP` *base* captioner + ST encoder        | Image → caption → embedding → semantic match                                 |
| **Both**   | Caption ✚ text                              | Captions and text concatenated before encoding                               |

* Adjustable **top-N** results (3 / 5 / 10 / 25).  
* Memory-mapped `.npy` keeps RAM usage low even for 100 k+ titles.  
* Runs identical on local Python, Docker and **Hugging Face Spaces** (auto-fetches data with `hf_hub_download`).  

---

## 📂 Repository layout

```
.
├─ app.py                 ← Gradio interface & API endpoints
├─ requirements.txt       ← Locked dependency versions
├─ data/                  ← Small example files for local dev
│   ├─ netflix_embeddings.npy
│   └─ netflix_metadata.csv
└─ assets/
    └─ screenshot.png
```

Large binaries are stored in a separate HF dataset repo: `your-username/netflix-rec-data`.

---

## 🚀 Quick start (local)

```bash
# 1. Clone and install deps (Python 3.10+)
python -m pip install -r requirements.txt

# 2. Run the demo
python app.py  # opens http://127.0.0.1:7860
```

If the two data files are missing, the app will download and cache them automatically via `huggingface_hub`.

---

## 🛰️ Deploy on Hugging Face Spaces

1. **Create a new Space → Gradio template**.  
2. Copy everything in this repo except `data/`.  
   (If you keep `data/`, consider Git LFS for the `.npy`.)  
3. In *Settings → Variables & Secrets* add a `HF_TOKEN` **if** the data repo is private.  
4. Push – the Space should build and launch automatically.

```python
# app.py (excerpt)
from huggingface_hub import hf_hub_download

data_repo = "your-username/netflix-rec-data"
emb_path  = hf_hub_download(repo_id=data_repo, filename="netflix_embeddings.npy")
meta_path = hf_hub_download(repo_id=data_repo, filename="netflix_metadata.csv")
```

---

## 🛠️ Regenerating embeddings

```python
from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np

model   = SentenceTransformer("all-MiniLM-L6-v2")
meta_df = pd.read_csv("netflix_titles.csv")
emb     = model.encode(
    meta_df["title"] + ": " + meta_df["description"],
    convert_to_numpy=True,
    show_progress_bar=True
).astype(np.float32)

np.save("netflix_embeddings.npy", emb)
```

Upload both files to the data repo (or replace the ones in `data/`).

---

## 🤝 Contributing

1. Fork ➜ branch ➜ PR.  
2. Ensure `pre-commit run --all-files` passes (formatting, Ruff, etc.).  
3. For sizeable changes open an issue first to discuss.

---

## 📄 License

This project is licensed under the **MIT License** – see `LICENSE` for details.

Netflix data © Netflix, used here for educational/demo purposes under *fair use*.

---

## 🙏 Acknowledgements

* [Sentence-Transformers](https://www.sbert.net/) by UKP TU-Darmstadt & AI2.  
* [BLIP](https://github.com/salesforce/LAVIS) by Salesforce Research.  
* [Gradio](https://gradio.app/) for the lightning-fast demo UI.  
* The open-source community ❤️  
```
