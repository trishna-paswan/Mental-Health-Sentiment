from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# 🔥 Load model from Hugging Face (NOT local)
try:
    classifier = pipeline(
        "text-classification",
        model="trishnaa-paswan/stress-detection-model",
        tokenizer="trishnaa-paswan/stress-detection-model",
        device=-1  # CPU (safe for Render)
    )
except Exception as e:
    classifier = None
    print(f"Error loading model: {e}")

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
        cause = request.form.get("cause", "")
        detailed = request.form.get("detailed_reason", "").strip()

        if not text and not detailed:
            return render_template("index.html", prediction="Please write something first!", mood_color=mood_color)

        # ✅ CLEAN INPUT (no duration)
        final_input = f"{text}. {detailed}. Cause: {cause}".strip()

        # 🔮 Prediction
        result = classifier(final_input)[0]
        label_id = result['label']
        confidence = round(result['score'] * 100, 2)

        # 🎯 Label Mapping
        label_map = {
            "LABEL_0": {"word": "Anxiety", "icon": "🌪️", "color": "text-red-400"},
            "LABEL_1": {"word": "Bipolar Disorder", "icon": "🎭", "color": "text-red-400"},
            "LABEL_2": {"word": "Depression", "icon": "🌧️", "color": "text-red-400"},
            "LABEL_3": {"word": "Normal", "icon": "🌿", "color": "text-green-400"},
            "LABEL_4": {"word": "Personality Disorder", "icon": "🧩", "color": "text-red-400"},
            "LABEL_5": {"word": "Stress", "icon": "💼", "color": "text-red-400"},
            "LABEL_6": {"word": "Suicidal Thoughts", "icon": "🆘", "color": "text-red-600 font-bold"}
        }

        # 🔁 Smart override
        low_confidence = result['score'] < 0.8

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

        # 💬 Response
        if mood_word == "Normal":
            message = "You seem to be in a balanced state. Keep prioritizing your peace ✨"
        elif mood_word == "Suicidal Thoughts":
            message = "URGENT: Please reach out to a professional or helpline immediately 💜"
        elif mood_word == "Stress":
            message = "You're carrying a lot. Take small breaks and breathe 💼"
        else:
            message = f"It sounds like you're experiencing {mood_word.lower()}. Be gentle with yourself 💜"

        # 📊 Insight
        insight = f"Context: {cause}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message,
        insight=insight,
        mood_color=mood_color,
        user_text=text if request.method == "POST" else "",
        user_detailed=detailed if request.method == "POST" else "",
        user_cause=cause if request.method == "POST" else ""
    )

if __name__ == "__main__":
    app.run(debug=True)