from flask import Flask, request, jsonify, render_template
import requests
import random
import os

app = Flask(__name__)

# Enable CORS
from flask_cors import CORS
CORS(app)

# HuggingFace API
API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

# Suggestions
suggestions = {
    "joy": [
        "Great to see you feeling positive! Keep journaling what made you happy today.",
        "This is wonderful! Share your joy with someone close to you.",
        "Celebrate this moment! What made you happiest today?",
        "Ride this wave of happiness—it's contagious! Spread it around."
    ],
    "sadness": [
        "It's okay to feel sad. Try a short walk outside or call someone you trust.",
        "Consider talking to someone about what's troubling you.",
        "Sadness is temporary. Give yourself permission to feel it.",
        "A warm cup of tea, some journaling, or a favorite song might help right now."
    ],
    "anger": [
        "Take 5 slow deep breaths. A short walk or exercise can help release tension.",
        "Channel this energy into something productive—physical activity works wonders.",
        "Step back for a moment. What's really bothering you beneath the surface?",
        "Try cold water on your face or intense exercise to calm the nervous system."
    ],
    "fear": [
        "Try the 5-4-3-2-1 grounding technique: name 5 things you can see right now.",
        "Remember: fear is just your mind trying to protect you. You're safe.",
        "Talk through your fears with someone you trust or write them down.",
        "Take deep breaths and remind yourself of times you've overcome challenges before."
    ],
    "neutral": [
        "You seem balanced. A short mindfulness session can help maintain that calm.",
        "This equilibrium is valuable—keep nurturing what brings you peace.",
        "Consider doing something you enjoy to enhance this positive state.",
        "Reflect on what's helping you stay grounded right now."
    ],
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please enter some text"}), 400

        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()

        print("API RESPONSE:", result)

        # If API sends error
        if isinstance(result, dict):
            return jsonify({"error": result.get("error", "API issue")}), 500

        if not isinstance(result, list) or not result:
            return jsonify({"error": "Invalid API response"}), 500

        top_emotion = max(result[0], key=lambda x: x["score"])
        emotion = top_emotion["label"].lower()
        score = round(top_emotion["score"] * 100, 1)

        if emotion not in suggestions:
            emotion = "neutral"

        emotion_mapping = {
            "disgust": "anger",
            "surprise": "neutral"
        }
        emotion = emotion_mapping.get(emotion, emotion)

        suggestion = random.choice(
            suggestions.get(emotion, ["Take care of yourself today."])
        )

        return jsonify({
            "emotion": emotion,
            "confidence": score,
            "suggestion": suggestion
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Something went wrong"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)