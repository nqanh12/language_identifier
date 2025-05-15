# Language Detection System

A machine learning-based system that can detect 7 different languages:
- Vietnamese
- English
- French
- Japanese
- German
- Spanish
- Korean

## Features
- Real-time language detection
- File-based detection (supports TXT, DOCX, PDF)
- Batch detection for multiple texts
- History tracking
- Probability visualization
- Feature importance analysis
- PDF report export
- Dark/Light theme support
- Training data management

## Requirements
- Python 3.8+
- Required packages listed in requirements.txt

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/language_identifier.git
cd language_identifier
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main application:
```bash
python main.py
```

2. To generate training data:
```bash
python generate_data.py
```

3. To train the model:
```bash
python train_model.py
```

## Project Structure
- `main.py`: Main application with GUI
- `generate_data.py`: Training data generation
- `train_model.py`: Model training script
- `data/`: Training data for each language
- `model/`: Saved model and vectorizer
- `reports/`: Generated reports

## License
MIT License

## Author
Nguyễn Quốc Anh
