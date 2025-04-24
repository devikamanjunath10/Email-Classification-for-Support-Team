import re
from collections import OrderedDict
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean and lowercase the input text.
    Removes extra spaces and strips leading/trailing spaces.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """
    Mask Personally Identifiable Information (PII) in the input text.
    
    Args:
        text (str): The input email body.
    Returns:
        tuple[str, list[dict]]: Masked text and list of masked entities.
    """
    entities = []
    replacements = []
    original_text = text
    text = text.lower()

    # Regex patterns for various PII
    patterns = [
        (r"\b\d{16}\b", "credit_debit_no", "[credit_debit_no]"),
        (r"\b\d{12}\b", "aadhar_num", "[aadhar_num]"),
        (r"\b\d{3}\b", "cvv_no", "[cvv_no]"),
        (r"\b\d{2}/\d{2}/\d{4}\b", "dob", "[dob]"),
        (r"\b\d{2}-\d{2}-\d{4}\b", "dob", "[dob]"),
        (r"\b\d{10}\b", "phone_number", "[phone_number]"),
        (
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "email",
            "[email]",
        ),
        (r"(?<=my name is )([a-z]+ [a-z]+)", "full_name", "[full_name]"),
        (r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b", "full_name", "[full_name]"),
        (r"\b(0[1-9]|1[0-2])/\d{2}\b", "expiry_no", "[expiry_no]"),
    ]

    for pattern, label, mask in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            original = original_text[start:end]
            replacements.append((start, end, label, mask, original))

    # Apply spaCy-based NER
    doc = nlp(original_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            original = original_text[start:end]
            replacements.append((start, end, "full_name", "[full_name]", original))
        elif ent.label_ in ["ORG", "GPE", "LOC"]:
            start, end = ent.start_char, ent.end_char
            original = original_text[start:end]
            replacements.append(
                (start, end, ent.label_, f"[{ent.label_.lower()}]", original)
            )

    replacements.sort(key=lambda x: x[0])
    masked_text = ""
    last_index = 0
    final_entities = []

    for start, end, label, mask, original in replacements:
        if start < last_index:
            continue
        masked_text += original_text[last_index:start] + mask

        entity_dict = OrderedDict()
        entity_dict["position"] = [start, end]
        entity_dict["classification"] = label
        entity_dict["entity"] = original
        final_entities.append(entity_dict)

        last_index = end

    masked_text += original_text[last_index:]
    return masked_text, final_entities
