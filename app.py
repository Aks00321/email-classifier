import re
import pandas as pd
import joblib
import spacy
import gradio as gr

nlp = spacy.load("en_core_web_sm")

# Load model/vectorizer
model = joblib.load('email_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_number": r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b',
    "dob": r'\b(?:\d{1,2}[/-])?(?:\d{1,2}[/-])?\d{2,4}\b',
    "aadhar_num": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    "credit_debit_no": r'\b(?:\d[ -]*?){13,16}\b',
    "cvv_no": r'\b\d{3}\b',
    "expiry_no": r'\b(0[1-9]|1[0-2])\/?([0-9]{2})\b'
}

def mask_pii(email):
    masked = email
    entities = []

    for label, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, email):
            start, end = match.start(), match.end()
            original = email[start:end]
            masked = masked.replace(original, f"[{label}]")
            entities.append({
                "position": [start, end],
                "classification": label,
                "entity": original
            })

    doc = nlp(email)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            original = email[start:end]
            if original not in masked:
                masked = masked.replace(original, "[full_name]")
                entities.append({
                    "position": [start, end],
                    "classification": "full_name",
                    "entity": original
                })

    return masked, entities

def classify_email(email_text):
    masked_email, masked_entities = mask_pii(email_text)
    email_vec = vectorizer.transform([masked_email])
    category = model.predict(email_vec)[0]

    return {
        "input_email_body": email_text,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, placeholder="Paste support email here..."),
    outputs="json",
    title="Akaike Email Classifier",
    description="Classifies support emails and masks PII (name, email, phone, etc.)."
).launch()
