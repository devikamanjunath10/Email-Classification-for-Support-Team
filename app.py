from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import gradio as gr  # Import Gradio
from utils import clean_text, mask_pii

# Initialize the Flask application and CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

# Load the pre-trained model, vectorizer, and category mapping
model = joblib.load('model/email_classifier_model.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
id2category = json.load(open('api/category_mapping.json'))

def classify_email_gradio(email_text: str) -> dict:
    """
    Function to classify the email and return the masked result using Gradio.

    Args:
        email_text (str): The raw input email text to be classified.

    Returns:
        dict: A dictionary containing the original email, masked email, 
              the classified category, and the list of masked entities.
    """
    # Clean and mask the email
    cleaned_email = clean_text(email_text)
    masked_email, entity_list = mask_pii(cleaned_email)

    # Vectorize the masked email text
    email_vectorized = vectorizer.transform([masked_email])

    # Predict the category of the email using the pre-trained model
    predicted_label = model.predict(email_vectorized)[0]
    predicted_class = id2category.get(str(predicted_label), "unknown")

    # Prepare the response
    response = {
        "input_email_body": email_text,
        "list_of_masked_entities": entity_list,
        "masked_email": masked_email,
        "category_of_the_email": predicted_class
    }

    return response

@app.route('/predict', methods=['POST'])
def classify_email() -> dict:
    """
    API endpoint to classify emails and mask PII.

    This function processes the incoming POST request, performs PII masking, 
    classifies the email, and returns a response with masked email and classification.

    Returns:
        dict: JSON response containing the original email, masked email, 
              masked entities, and classification.
    """
    try:
        # Get the incoming request data
        data = request.get_json()

        # Check if the 'email' key is present in the request data
        if 'email' not in data:
            return jsonify({
                "error": "Missing 'email' key in the request data"
            }), 400

        email_text = data['email']

        # Clean and mask the email
        cleaned_email = clean_text(email_text)
        masked_email, entity_list = mask_pii(cleaned_email)

        # Vectorize the masked email text
        email_vectorized = vectorizer.transform([masked_email])

        # Predict the category of the email using the pre-trained model
        predicted_label = model.predict(email_vectorized)[0]
        predicted_class = id2category.get(str(predicted_label), "unknown")

        # Prepare the response
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
        # Handle missing keys in the input data
        return jsonify({
            "error": f"Missing key: {str(e)}"
        }), 400

    except Exception as e:
        # Handle any other errors
        return jsonify({
            "error": str(e)
        }), 500

def gradio_interface() -> None:
    """
    Launches the Gradio interface for testing the email classifier and PII masker.

    This interface allows users to input an email, classify its category, 
    and view the masked PII entities.
    """
    gr.Interface(
        fn=classify_email_gradio,
        inputs=gr.Textbox(lines=5, label="Input Email"),
        outputs=[gr.JSON(label="Response")],
        title="Email Classifier & PII Masker",
        description="Enter an email to classify its category and mask PII entities."
    ).launch()

if __name__ == '__main__':
    # Launch the Gradio interface for testing
    gradio_interface()  # Launch the Gradio UI for testing

    # Run the Flask app
    app.run(host="0.0.0.0", port=7860)