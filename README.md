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

### 🤖 ML-Powered Investment Scoring
- **XGBoost Price Prediction** — accurate property valuation models  
- **LightGBM Investment Classification** — investment potential assessment  
- **Isolation Forest Anomaly Detection** — identifies unusual market opportunities  
- **Ensemble Scoring** — combined investment scores with fallback heuristic method

### 🎛️ Interactive Dashboard
- **4-Tab Interface:** Recommendations, Market Analysis, Top Picks, Area Performance
- **Smart Filtering:** Area, property type, budget range, and investment score thresholds
- **Investment Preferences:** Customizable horizon (1-2, 3-5, 5+ years) and risk tolerance
- **Interactive Visualizations:** Plotly charts with scatter plots, histograms, and area comparisons

### 📊 Advanced Analytics
- **Feature Engineering:** 20+ metrics including developer reputation, location premiums, market velocity
- **Quality Filtering:** Only properties with complete data and sufficient transaction history
- **Risk Assessment:** Automated labeling based on ML predictions and market indicators

## 🎬 Demo

Watch the application in action! The demo video showcases all the key features including the ML-powered recommendations, interactive dashboard, and filtering capabilities.

📹 **[Download Demo Video](https://github.com/rajrejin/Dubai-Property-Recommender/raw/main/demo/Property%20Reccomender%20Demo.mp4)**

The demo showcases:
- AI-powered investment scoring using XGBoost, LightGBM, and Isolation Forest models
- Interactive property filtering by area, type, budget range, and investment score
- Multi-tab dashboard with Recommendations, Market Analysis, Top Picks, and Area Performance
- Real-time investment score calculations with risk assessment labels
- Plotly visualizations including scatter plots, histograms, and area performance charts
- Personalized recommendations based on investment horizon and risk tolerance
- CSV export functionality for filtered investment opportunities

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
- Python 3.7+ and a modern web browser
- 4GB+ RAM for dataset processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajrejin/Dubai-Property-Recommender.git
   cd Dubai-Property-Recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Get the official dataset from [Dubai Pulse - DLD Transactions](https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open)
   - Save as `data/Transactions.csv` in the project folder

4. **Run the application**
   ```bash
   streamlit run property_recommender_app.py
   ```

5. **Access the dashboard**
   - Open your browser to the displayed local address (typically http://localhost:8501)
   - Wait for initial data processing and model training (1-2 minutes)

## 📊 Data Source

**Dubai Land Department** transaction data from [Dubai Pulse](https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open) — the official open data platform of Dubai Government.

**Key Attributes:** Property details, transaction history, location data, developer information, and proximity to amenities (metro, malls).

## 🔧 Technical Architecture

### Data Processing Pipeline
1. **Data Loading & Cleaning** — Dubai Pulse CSV processing with quality filters
2. **Feature Engineering** — 20+ metrics including time-based, area-level, and developer features
3. **ML Training** — XGBoost, LightGBM, Isolation Forest on representative samples
4. **Scoring & Ranking** — Ensemble ML scoring with heuristic fallback

### Performance & Observability
- Model training uses representative samples (up to 30K properties from top 50 areas)
- Live metrics displayed during training for transparency
- Graceful fallback to heuristic scoring if ML libraries unavailable

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

- **Python 3.7+** with standard data science libraries
- **Core Dependencies:** streamlit, pandas, numpy, plotly
- **ML Libraries (Optional):** xgboost, lightgbm, scikit-learn
- **System:** 4GB+ RAM recommended for dataset processing

> **Note:** ML libraries are optional - the app includes a sophisticated heuristic fallback that provides similar functionality without ML dependencies.

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

*Last Updated: September 2025*