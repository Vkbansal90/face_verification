from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

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

