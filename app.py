from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import threading
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Preload models on startup in a background thread ---
def preload_models():
    try:
        logging.info("Preloading DeepFace models (this can take a while)...")
        # Build the face recognition model used by DeepFace (e.g., Facenet)
        # Change model name if you prefer another model: "VGG-Face", "ArcFace", etc.
        DeepFace.build_model("Facenet")
        # Preload detector models indirectly by calling verify with two tiny images if needed,
        # but DeepFace will load detector lazily on first verify call. The build_model above helps.
        logging.info("DeepFace models preloaded.")
    except Exception:
        logging.exception("Failed to preload DeepFace models.")

# Start preloading in a background thread so app can start quickly
t = threading.Thread(target=preload_models, daemon=True)
t.start()
# ------------------------------------------------------

@app.route('/')
def home():
    return "Face Verification API is running!"

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    try:
        img1 = request.files['image1']
        img2 = request.files['image2']

        img1_path = "temp1.jpg"
        img2_path = "temp2.jpg"
        img1.save(img1_path)
        img2.save(img2_path)

        result = DeepFace.verify(img1_path, img2_path)

        os.remove(img1_path)
        os.remove(img2_path)

        return jsonify({
            "match": bool(result["verified"]),
            "distance": result["distance"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
