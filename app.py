from flask import Flask, render_template, request, jsonify
from recommender import Recommender
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "products.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "data", "ratings.csv")

recommender = Recommender(DATA_PATH, RATINGS_PATH)


@app.route("/")
def index():
    products = recommender.list_products()
    return render_template("index.html", products=products)


@app.route("/api/recommend")
def api_recommend():
    product_id = request.args.get("product_id", "")
    mode = request.args.get("mode", "hybrid")
    try:
        n = int(request.args.get("n", "5"))
    except:
        n = 5
    try:
        content_w = float(request.args.get("content_w", "0.5"))
    except:
        content_w = 0.5

    recs = recommender.recommend(product_id, top_n=n, mode=mode, content_w=content_w)
    return jsonify(recs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
