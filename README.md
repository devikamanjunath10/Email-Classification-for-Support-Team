
# Email-Classification-for-Support-Team

this project aims to design and implement an automated email classification system for a company’s support team. The system is intended to categorize incoming support emails into predefined categories such as Billing Issues, Technical Support, and Account Management. A critical requirement is ensuring that all personally identifiable information (PII) is accurately detected and masked before the email is processed for classification. Once the email is categorized, the original data must be securely restored (demasked) to maintain data integrity. This ensures both compliance with data privacy standards and efficient handling of support queries
## Project Overview
This project implements a robust end-to-end system for automatically classifying support emails while preserving user privacy through PII masking and restoration. It begins by detecting personally identifiable information using a hybrid approach combining regular expressions and spaCy’s Named Entity Recognition (NER). Entities such as names, email addresses, phone numbers, Aadhaar numbers, and payment details are masked to prevent exposure during processing. The masked email is then classified using a Multinomial Naive Bayes model trained to categorize emails into predefined support categories like Billing Issues, Technical Support, and Account Management. Finally, the original sensitive information is accurately restored, ensuring both secure handling and meaningful communication in the output.
## Setup and Intallation
1. Clone the Repository
```
git clone https://github.com/your-username/email-support-classifier.git

cd email-support-classifier

```
2. Create and Activate Virtual Environment
```
python -m venv venv
source venv/bin/activate
```
3. Install Required Dependencies
```
pip install -r requirements.txt
```
4. Download spaCy Model
```
python -m spacy download en_core_web_sm
```
## Usage Instruction
Usage Instructions
This project masks Personally Identifiable Information (PII) from user-provided text such as emails or support tickets. It uses regular expressions and spaCy's NLP capabilities to detect and replace sensitive data with placeholder tags.
 
## 1.Deploying on GitHub

To deploy the project on GitHub, follow these steps:

**1.Create a new GitHub repository**:
   - Go to [GitHub](https://github.com) and create a new repository (e.g., `email-support-classifier`).

**2.Clone the repository**:
   ```bash
   git clone https://github.com/your-username/email-support-classifier.git
   cd email-support-classifier
   ```
**3.Push your project files to the repository:**
```
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/email-support-classifier.git
git push -u origin main
```
**4.Run your app locally or deploy it on a server**

## 2.Deploying on Hugging face space

**1.Create a new Hugging Face Space**

**2.Upload your project files**

**3.Ensure app.py has the Gradio interface**

**4.Run your Hugging Face Space:**

- Hugging Face will automatically run your app once the files are uploaded!


