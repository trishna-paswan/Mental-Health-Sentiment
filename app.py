from flask import Flask, render_template, request
from transformers import pipeline
import os

app = Flask(__name__)

# Load DistilBERT model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "distilbert_model")

try:
    # Using pipeline for easy inference
    classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
except Exception as e:
    classifier = None
    print(f"Error loading DistilBERT: {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = None
    message = ""
    insight = ""
    mood_color = "text-white"

    if request.method == "POST":
        if classifier is None:
            return render_template("index.html", prediction="Error: Model not loaded", mood_color=mood_color)

        text = request.form.get("text", "").strip()
        duration = request.form.get("duration", "")
        cause = request.form.get("cause", "")
        detailed = request.form.get("detailed_reason", "").strip()

        if not text and not detailed:
            return render_template("index.html", prediction="Please write something first!", mood_color=mood_color)

        # ✅ Use ALL inputs as requested
        final_input = f"{text} {detailed} {duration} {cause}".strip()

        # 🔮 Predict using DistilBERT
        result = classifier(final_input)[0]
        label_id = result['label']
        confidence = round(result['score'] * 100, 2)

        # 🎯 Refined Mental Health Mapping (7-Class DistilBERT)
        # 0: Anxiety, 1: Bipolar, 2: Depression, 3: Normal, 4: Personality, 5: Stress, 6: Suicidal
        label_map = {
            "LABEL_0": {"word": "Anxiety", "icon": "🌪️", "color": "text-red-400"},
            "LABEL_1": {"word": "Bipolar Disorder", "icon": "🎭", "color": "text-red-400"},
            "LABEL_2": {"word": "Depression", "icon": "🌧️", "color": "text-red-400"},
            "LABEL_3": {"word": "Normal", "icon": "🌿", "color": "text-green-400"},
            "LABEL_4": {"word": "Personality Disorder", "icon": "🧩", "color": "text-red-400"},
            "LABEL_5": {"word": "Stress", "icon": "💼", "color": "text-red-400"},
            "LABEL_6": {"word": "Suicidal Thoughts", "icon": "🆘", "color": "text-red-600 font-bold"}
        }
        
        # Keyword-based override for better "satisfaction"
        # If the model is unsure (LABEL_3 is often a catch-all), we check for strong keywords
        low_confidence = result['score'] < 0.8
        detected_word = label_map.get(label_id, {"word": label_id})["word"]
        
        if label_id == "LABEL_3" or low_confidence:
            check_text = final_input.lower()
            if any(w in check_text for w in ["stress", "overwhelmed", "workload", "pressure"]):
                label_id = "LABEL_5"
            elif any(w in check_text for w in ["depressed", "sad", "unhappy", "lonely", "hopeless"]):
                label_id = "LABEL_2"
            elif any(w in check_text for w in ["anxious", "panic", "worry", "nervous"]):
                label_id = "LABEL_0"
            elif any(w in check_text for w in ["die", "kill", "end it", "suicide"]):
                label_id = "LABEL_6"

        mapped_data = label_map.get(label_id, {"word": label_id, "icon": "🔍", "color": "text-white"})
        mood_word = mapped_data["word"]
        mood_icon = mapped_data["icon"]
        mood_color = mapped_data["color"]
        
        prediction = f"{mood_icon} {mood_word}"
        
        # 💬 Smarter, more empathetic responses
        if mood_word == "Healthy/Normal":
            message = "You seem to be in a balanced state. It's great to check in on yourself! Keep prioritizing your peace ✨"
        elif mood_word == "Suicidal Thoughts":
            message = "URGENT: Your life is incredibly valuable. Please reach out to a professional or a crisis helpline immediately. Help is available 24/7 💜"
        elif mood_word == "Stress":
            message = "You're carrying a heavy load. Remember to take small breaks and breathe. You don't have to do it all at once 💼"
        else:
            message = f"It sounds like you're experiencing {mood_word.lower()}. Be gentle with yourself today—you're doing your best 💜"

        # 📊 Insight
        insight = f"Context: {cause} | Timeframe: {duration}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message,
        insight=insight,
        mood_color=mood_color,
        user_text=text if request.method == "POST" else "",
        user_detailed=detailed if request.method == "POST" else "",
        user_duration=duration if request.method == "POST" else "",
        user_cause=cause if request.method == "POST" else ""
    )

if __name__ == "__main__":
    app.run(debug=True)
