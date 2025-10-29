# E-commerce Hybrid Recommender (DNA project)

This project is an upgraded e-commerce recommender demonstrating a **hybrid approach**:
- **Content-based filtering** using TF-IDF on product titles + descriptions.
- **Item-based collaborative filtering** using user ratings (cosine similarity on item rating vectors).
- An API that supports `mode=content`, `mode=collab`, or `mode=hybrid` (weighted combination).

## Files
- `app.py` — Flask web app and API.
- `recommender.py` — Recommender implementation (content, collab, hybrid).
- `data/products.csv` — Expanded product list (~25 items).
- `data/ratings.csv` — Simulated user ratings (user_id,product_id,rating).
- `templates/index.html`, `static/style.css` — Frontend.
- `requirements.txt` — Python dependencies.

## Run locally
1. Create and activate virtual environment (recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Open: `http://127.0.0.1:5000`

## API examples
- Content-based: `/api/recommend?product_id=1&mode=content&n=5`
- Collaborative: `/api/recommend?product_id=1&mode=collab&n=5`
- Hybrid: `/api/recommend?product_id=1&mode=hybrid&n=5&content_w=0.6`

