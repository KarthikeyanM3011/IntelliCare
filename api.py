from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)
#
# Configure the Generative AI API key
GOOGLE_API_KEY = 'Your_Api_Key'
genai.configure(api_key=GOOGLE_API_KEY)

# Set the model name
model_name = 'gemini-1.0-pro'

# Initialize the Generative AI model
if model_name:
    model = genai.GenerativeModel(model_name)
else:
    raise Exception("No model found that supports 'generateContent'")

@app.route('/generate_content', methods=['POST'])
def generate_content():
    data = request.json
    input_text = data.get('input_text', '')

    if not input_text:
        return jsonify({'error': 'No input_text provided'}), 400

    try:
        output = model.generate_content(input_text)
        return jsonify({'generated_content': output.text})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == 'main':
    print("Running")
    app.run(port=5000, debug=True)
