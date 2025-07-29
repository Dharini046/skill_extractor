from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load your fine-tuned FLAN-T5 model
MODEL_PATH = "./flan-t5-skill-extraction-final"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get("prompt", "")

    if not input_text:
        return jsonify({"error": "No prompt provided"}), 400

    full_prompt = f"Extract skills: {input_text}"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=64)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"skills": result})

if __name__ == '__main__':
    app.run(debug=True)
    