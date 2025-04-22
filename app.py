from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
from utils import clean_text, mask_pii

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load('model/email_classifier_model.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
id2category = json.load(open('api/category_mapping.json'))


@app.route('/predict', methods=['POST'])
def classify_email():
    """
    API endpoint to classify emails and mask PII.
    """
    try:
        data = request.get_json()
        if 'email' not in data:
            return jsonify({
                "error": "Missing 'email' key in the request data"
            }), 400

        email_text = data['email']

        # Clean and mask the email
        cleaned_email = clean_text(email_text)
        masked_email, entity_list = mask_pii(cleaned_email)

        # Vectorize the masked email
        email_vectorized = vectorizer.transform([masked_email])

        # Prediction
        predicted_label = model.predict(email_vectorized)[0]
        predicted_class = id2category.get(str(predicted_label), "unknown")

        # Prepare the response (in correct JSON key order)
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": entity_list,
            "masked_email": masked_email,
            "category_of_the_email": predicted_class
        }

        return app.response_class(
            response=json.dumps(response),
            status=200,
            mimetype='application/json'
        )

    except KeyError as e:
        return jsonify({
            "error": f"Missing key: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)
