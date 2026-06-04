# IME — Maritime Intelligence Platform
## Enterprise Edition with ML Classifier & Match Engine

### Quick Start

```bash
pip install flask scikit-learn numpy
python app.py
```
Open http://localhost:5050

### First Run
1. Click the **⬆ Load Sample Data** button in the top bar
2. This seeds the database with real maritime emails AND trains the ML model
3. Navigate to any page to explore

### Features

| Feature | Description |
|---------|-------------|
| Command Center | Live operations dashboard with world map & email stream |
| Email Processing | Paste any maritime email → AI extracts structured data |
| ML Classifier | TF-IDF + Logistic Regression, ~95% accuracy, confusion matrix |
| Match Engine | Vessel ↔ Cargo matching by region/DWT/laycan |
| Analytics | Executive charts, market activity, performance metrics |
| Vessel Database | All open tonnage positions |
| Cargo VC/TC | Voyage & time charter cargo database |

### ML Classifier
- **Algorithm**: TF-IDF Vectorizer + Logistic Regression (scikit-learn)
- **Accuracy**: ~95% on cross-validation
- **Training data**: 24 built-in samples + all processed emails
- **Features**: Bigram TF-IDF, 3000 vocabulary
- **Output**: Per-class probabilities, confusion matrix, top features

### Match Engine
- Matches open vessels to cargo by:
  - Regional proximity (12 maritime regions)
  - DWT compatibility
  - Laycan/open-date overlap
  - Cargo type suitability
- Scoring: 0-100 confidence score
- Supports both VC and TC cargo types
