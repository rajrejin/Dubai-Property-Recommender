# 🏢 Dubai Property Recommender

An intelligent property investment recommendation system for Dubai real estate market, powered by machine learning models trained on historical transactions with strong classifier performance observed during internal evaluation.

![Dubai Real Estate](https://img.shields.io/badge/Dubai-Real%20Estate-blue) ![ML Powered](https://img.shields.io/badge/ML-Powered-green) ![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)

## 🎯 Overview

This application provides data-driven investment recommendations for Dubai's property market using sophisticated machine learning algorithms. It processes official transaction data from the Dubai Land Department to identify optimal investment opportunities based on price efficiency, market trends, and risk assessment.

> **Implementation notes (consistent with the code):**
> - The app loads the official `Transactions.csv` from Dubai Pulse for analytics and area-level intelligence.
> - For in-app model training, the code trains ML models on a representative sample for performance reasons.
> - If the ML libraries (XGBoost / LightGBM / scikit-learn) are not available, the app falls back to an advanced heuristic scoring method so the UI remains functional.

## ✨ Key Features

### 🤖 Advanced ML Scoring Engine
- **XGBoost Price Prediction** — property valuation models  
- **LightGBM Investment Probability** — investment potential assessment  
- **Isolation Forest Anomaly Detection** — identifies unusual market opportunities  
- **Ensemble Scoring** — combined investment scores for ranking

> Note: classifier performance was observed on representative training splits during evaluation; exact results are shown live in the app after training.

### 📊 Intelligent Analytics
- Many engineered features capturing developer reputation, location premiums, and market velocity  
- Quality data filtering to keep only actionable properties with complete information  
- Area performance insights and risk classification

### 🎨 Interactive Dashboard
- Multi-tab interface: Recommendations, Area Analysis, Charts  
- Real-time filtering by area, property type, budget, and score thresholds  
- Interactive Plotly visualizations and CSV export for filtered results

### 🔍 Smart Filtering System
- Geographic filtering with high-quality areas only  
- Property type filtering (apartments, villas, offices, shops, etc.)  
- Customizable budget and score-based thresholds

## 🎬 Demo

Watch the application in action! The demo video showcases all the key features including the ML-powered recommendations, interactive dashboard, and filtering capabilities.

📹 **[View Demo Video](./demo/Property%20Reccomender%20Demo.mp4)**

The demo covers:
- Setting up and launching the application
- Exploring the recommendation engine
- Using the interactive filtering system
- Analyzing area performance metrics
- Exporting investment recommendations

## 📁 Project Structure

```
Dubai-Property-Recommender/
├── property_recommender_app.py    # Main Streamlit application
├── data/
│   └── Transactions.csv           # Downloaded dataset (from Dubai Pulse; large — keep out of repo)
├── demo/
│   └── Property Reccomender Demo.mp4  # Application demo video
├── requirements.txt               # Python dependencies
├── LICENSE                        # Project license
└── README.md                      # This file
```

> **Important:** `Transactions.csv` is maintained on Dubai Pulse and updated regularly. The app expects this file at `data/Transactions.csv`. Do **not** commit large raw datasets to the repository — host externally or use Git LFS if necessary.

## 🚀 Quick Start

### Prerequisites
- Python and a modern web browser  
- Sufficient RAM and disk space to work with the official dataset

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Dubai-Property-Recommender"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Get the official dataset from Dubai Pulse: [Dubai Pulse - DLD Transactions](https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open)  
   - Save the CSV as `data/Transactions.csv` in this project folder.  
   - The app will load the full CSV for analytics and automatically use a representative sample for any in-app model training.

4. **Run the application**
   ```bash
   streamlit run property_recommender_app.py
   ```

5. **Access the dashboard**
   - Open your browser to the local Streamlit address and wait for initial processing. Model training, if run, performs training on a representative sample and will display the observed metrics in the UI.

## 📊 Data Source

The application uses official Dubai Land Department transaction data sourced from **Dubai Pulse**, the open data platform of the Dubai Government.

- **Data Portal**: Dubai Pulse (DLD Transactions)  
- **File Name**: `Transactions.csv` (updated regularly on the portal)  
- The code uses the full dataset for area-level analytics and a representative sample for in-app training.

### Key Data Attributes Used
- Property details: type, sub-type, area, project name  
- Transaction info: date, price, area size, price per square meter  
- Location info: area name, nearest metro, nearest mall  
- Developer info: project details and registration type

## 🔧 Technical Architecture

### Machine Learning Pipeline
1. **Data Preprocessing**
   - Quality filtering, outlier handling, and feature engineering

2. **Model Training**
   - XGBoost for price regression  
   - LightGBM for investment classification  
   - Isolation Forest for anomaly detection  
   - Training is performed on a representative sample in-app; the code falls back to a heuristic scoring method if ML libraries are absent

3. **Feature Engineering**
   - Area-level market intelligence, developer reputation metrics, location premiums, market velocity, and price efficiency ratios

### Performance & Observability
- Exact model metrics vary by run and hardware. The app displays observed training metrics after in-app training so users see reproducible, run-specific numbers.

## 🎛️ Application Features

### Recommendations Tab
- Top-ranked properties by model or heuristic score  
- Property detail table, risk labels, and export functionality

### Area Performance Tab
- Market performance by area and transaction volume analysis

### Interactive Charts
- Investment score scatter plots, price relationships, and volatility visualizations

### Sidebar Controls
- Area selection, property types, budget slider, and minimum score filter

## 💡 Investment Methodology

### Scoring Algorithm
The investment score combines multiple components including price efficiency, ML-derived probability, anomaly detection, and market liquidity. If ML models are unavailable, a heuristic scoring function mirrors these components.

### Quality Filters Applied
- Only high-quality, complete records are used  
- Areas with sufficient transaction data are prioritized

### Risk Assessment
- Investments are labeled based on model/heuristic outputs to indicate higher or lower expected suitability

## 🛠️ Customization

### Adding New Features
To extend analysis:
1. Modify feature engineering in `load_and_process_data()`  
2. Update the ML feature list in `prepare_ml_features()`  
3. Adjust the scoring logic in `score_properties_ml()` or the heuristic fallback

### Filtering Logic
Customize `filter_properties()` to introduce new attributes, budget logic, or custom thresholds.

## 📋 Requirements

### System Requirements
- Cross-platform (Windows, macOS, Linux)  
- Python 3 and relevant packages installed  
- Adequate RAM and disk space for dataset processing

### Python Dependencies
See `requirements.txt` for required packages. ML libraries are optional for full functionality; the app will run without them using the heuristic fallback.

## 🚨 Troubleshooting

### Common Issues
- Memory constraints: consider using a smaller sample for development  
- Slow initial load: preprocessing can be performed once and cached for subsequent runs  
- Missing dependencies: install via the requirements file

## 🔮 Future Enhancements

### Planned Features
- Real-time integration, portfolio management, time-series forecasting, mobile responsiveness, and user authentication

### Advanced Analytics
- Comparative market analysis, ROI projections, market cycle insights, and developer performance tracking

## 📞 Support

For support or feature requests, open an issue in the repository and include environment details and any error messages.

## 📄 License

This project is for educational and research purposes. Underlying data is provided by Dubai Government under their open data policies.

## 🙏 Acknowledgments

- Dubai Land Department  
- Dubai Pulse  
- Streamlit community and ML library authors

---

**Built with ❤️ for Dubai's Real Estate Investment Community**

*Last Updated: August 2025*