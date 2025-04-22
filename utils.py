import re
from collections import OrderedDict
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Lowercase, normalize whitespace, and remove extra characters.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """
    Mask PII using spaCy NER and regex.
    Returns masked text and entity metadata in order.
    """
    entities = []
    replacements = []

    # Regex-based PII patterns
    patterns = [
    (r"\b\d{16}\b", "credit_debit_no", "[credit_debit_no]"),  # 16-digit cards
    (r"\b\d{12}\b", "aadhar_num", "[aadhar_num]"),            # Aadhaar
    (r"\b\d{3}\b", "cvv_no", "[cvv_no]"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "dob", "[dob]"),
    (r"\b\d{2}/\d{2}\b", "expiry_no", "[expiry_no]"),
    (r"\b\d{10}\b", "phone_number", "[phone_number]"),
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "email", "[email]"),
    (r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b", "full_name", "[full_name]"),
     ]


    # Apply regex-based masking
    for pattern, label, mask in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            original = match.group()
            replacements.append((start, end, label, mask, original))

    # spaCy-based NER (for PERSON, ORG, GPE, LOC)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
            start, end = ent.start_char, ent.end_char
            original = text[start:end]
            label = ent.label_
            mask = f"[{label.lower()}]"
            replacements.append((start, end, label, mask, original))

    # Sort replacements by start index
    replacements.sort(key=lambda x: x[0])

    masked_text = ""
    last_index = 0
    final_entities = []

    # Apply the replacements in order
    for start, end, label, mask, original in replacements:
        if start < last_index:
            continue  # Skip overlapping entities
        masked_text += text[last_index:start] + mask

        entity_dict = OrderedDict()
        entity_dict["position"] = [start, end]
        entity_dict["classification"] = label
        entity_dict["entity"] = original

        final_entities.append(entity_dict)
        last_index = end

    masked_text += text[last_index:]
    return masked_text, final_entities


def demask_pii(masked_text: str, entities: list[dict]) -> str:
    """
    Restore original PII entities by matching custom placeholder tags.
    """
    classification_to_placeholder = {
        "email": "[email]",
        "phone_number": "[phone_number]",
        "full_name": "[full_name]",
        "dob": "[dob]",
        "aadhar_num": "[aadhar_num]",
        "credit_debit_no": "[credit_debit_no]",
        "cvv_no": "[cvv_no]",
        "expiry_no": "[expiry_no]",
        "PERSON": "[person]",
        "ORG": "[org]",
        "GPE": "[gpe]",
        "LOC": "[loc]"
    }

    restored_text = masked_text
    for ent in entities:
        classification = ent.get("classification", "")
        entity = ent.get("entity", "")
        placeholder = classification_to_placeholder.get(classification)
        if placeholder:
            restored_text = restored_text.replace(placeholder, entity, 1)

    return restored_text


if __name__ == "__main__":
    sample = ("Hi team, my name is jhon and my Date of birth is 10/06/2003 and my "
              "credit Card Number is 4339977243 and Aadhar card number is 23413253 and "
              "CVV number is 345 and card expiry number is 347434 please call me at 9876543210 "
              "or email john.doe@example.com urgently.")
    
    masked, ents = mask_pii(sample)
    print("Masked Text:", masked)
    print("Entities:", ents)
    print("Restored:", demask_pii(masked, ents))