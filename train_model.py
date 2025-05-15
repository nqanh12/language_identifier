# train_model.py
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from collections import defaultdict

def load_data():
    texts = []
    labels = []
    lang_map = {
        'vi': 'Vietnamese',
        'en': 'English',
        'fr': 'French',
        'jp': 'Japanese',
        'de': 'German',      
        'es': 'Spanish',     
        'ko': 'Korean'       
    }
    
    for lang_code, lang_name in lang_map.items():
        file_path = f'data/{lang_code}.txt'
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found")
            continue
            
        try:
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        texts.append(line)
                        labels.append(lang_name)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            
    if not texts:
        raise ValueError("No training data found")
        
    return texts, labels


def get_top_features(vectorizer, model, n=20):
    """Get top N features for each language"""
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = defaultdict(list)
    
    for i, lang in enumerate(model.classes_):
        # Get feature importance for this class
        importance = model.feature_log_prob_[i]
        # Get indices of top N features
        top_indices = np.argsort(importance)[-n:]
        # Get feature names and their importance
        for idx in top_indices:
            feature_importance[lang].append({
                'feature': feature_names[idx],
                'importance': float(importance[idx])
            })
    
    return feature_importance

# Generate report   
def generate_report(accuracy, report, feature_importance):
    """Generate markdown report"""
    report_content = """# Language Detection System Report

## Model Performance
- Accuracy: {:.2f}%

## Classification Report
```
{}

## Top Features by Language
""".format(accuracy * 100, report)

    for lang, features in feature_importance.items():
        report_content += f"\n### {lang}\n"
        for feat in features:
            report_content += f"- {feat['feature']}: {feat['importance']:.2f}\n"

    # Add theoretical explanation
    report_content += """
## Theoretical Background

### Multinomial Naive Bayes
The Multinomial Naive Bayes classifier is particularly suitable for text classification tasks. It works by:
1. Calculating the probability of each word occurring in each language
2. Using these probabilities to determine the most likely language for a given text
3. Taking into account the frequency of words (bag-of-words approach)

### Role of Word Frequency
- Word frequency plays a crucial role in language detection
- Common words and their patterns are unique to each language
- The model learns these patterns during training
- More training data leads to better recognition of language-specific patterns
"""

    return report_content

try:
    # Load and prepare data
    print("Loading training data...")
    texts, labels = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Create vectorizer and model
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,  
        min_df=2,  
        token_pattern=r'(?u)\b\w+\b',
        analyzer='char_wb'  
    )
    
    model = MultinomialNB(alpha=0.01) 
    
    # Transform and fit
    print("Training model...")
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    X_test_vec = vectorizer.transform(X_test)
    test_accuracy = model.score(X_test_vec, y_test)
    print(f"Accuracy: {test_accuracy:.3f}")
    
    # Get predictions and classification report
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Get top features
    feature_importance = get_top_features(vectorizer, model)
    
    # Generate and save report
    report_content = generate_report(test_accuracy, report, feature_importance)
    os.makedirs('reports', exist_ok=True)
    with open('reports/model_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/nb_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    # Save feature importance
    with open('model/feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, ensure_ascii=False, indent=2)
    
    print("All files saved successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
