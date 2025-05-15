import re
import unicodedata
from typing import List, Dict, Tuple

def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Normalizing unicode characters
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    return text

def evaluate_model(model, vectorizer, X_test, y_test) -> Tuple[float, Dict]:
    """
    Evaluate model performance and return accuracy and detailed metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    # Transform test data
    X_test_vec = vectorizer.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate accuracy
    accuracy = model.score(X_test_vec, y_test)
    
    # Get detailed metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def get_language_info(lang_code: str) -> Dict:
    """
    Get information about supported languages
    """
    lang_info = {
        'vi': {
            'name': 'Vietnamese',
            'script': 'Latin',
            'family': 'Austroasiatic'
        },
        'en': {
            'name': 'English',
            'script': 'Latin',
            'family': 'Indo-European'
        },
        'fr': {
            'name': 'French',
            'script': 'Latin',
            'family': 'Indo-European'
        },
        'jp': {
            'name': 'Japanese',
            'script': 'Mixed (Kanji, Hiragana, Katakana)',
            'family': 'Japonic'
        }
    }
    return lang_info.get(lang_code, {})
