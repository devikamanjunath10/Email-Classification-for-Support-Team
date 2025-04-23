
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
