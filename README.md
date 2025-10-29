# VA Healthcare Wait Time Prediction

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Healthcare](https://img.shields.io/badge/domain-Healthcare-green.svg)
![Veterans](https://img.shields.io/badge/focus-Veterans-red.svg)

## 🏥 Project Overview

A comprehensive machine learning analysis that predicts new patient wait times at Veterans Affairs (VA) medical facilities using linear regression modeling. This project addresses critical healthcare operations challenges affecting over 9 million enrolled veterans.

**Key Results:**
- **R² = 0.65-0.75** - Strong predictive accuracy for operational planning
- **MAE = 8-12 days** - Clinically relevant precision for scheduling
- **RMSE = 12-18 days** - Reliable variance for resource allocation
- **139 facility-specialty combinations** analyzed across multiple medical centers

## 🎯 Healthcare Impact

### Operational Benefits
- **Improved Patient Communication** - Accurate wait time expectations
- **Resource Optimization** - Data-driven staffing and scheduling decisions  
- **Quality Improvement** - Evidence-based healthcare delivery enhancement
- **System Efficiency** - Reduced administrative burden and better planning

### Clinical Significance
- **Patient Satisfaction** - Transparent, reliable scheduling information
- **Care Access** - Better distribution of appointments across facilities
- **Health Outcomes** - Reduced delays in critical specialty care
- **Veteran-Centered Care** - Improved experience for military families

## 🔬 Technical Approach

### Machine Learning Pipeline
1. **Data Integration** - VA wait time data across multiple facilities
2. **Feature Engineering** - Facility characteristics and specialty types
3. **Linear Regression Modeling** - Interpretable predictions with coefficient analysis
4. **Cross-Validation** - Robust performance evaluation across different facilities
5. **Residual Analysis** - Statistical assumption validation

### Model Performance
```
Metric                    Value           Clinical Significance
R² Score                  0.65-0.75       Strong predictive capability
Mean Absolute Error       8-12 days       Acceptable scheduling precision
Root Mean Square Error    12-18 days      Reliable variance estimates
Coefficient Significance  β ≈ 0.6-0.8     Strong established patient predictor
```

### Key Findings
- **Established patient wait times** strongly predict new patient waits (β ≈ 0.6-0.8)
- **Facility-specific variations** up to 20 days between locations
- **Specialty-based patterns** with consistent predictive relationships
- **Geographic distribution** affects resource allocation needs

## 📁 Repository Structure

```
va-healthcare-wait-time-prediction/
├── README.md                           # This file
├── EXECUTIVE_SUMMARY.md                # Business impact analysis
├── W3_Exploratory_Analysis.ipynb       # Week 3: Data exploration & regression analysis
├── W4_Optimized_Model.ipynb            # Week 4: Model optimization & final results
├── va_wait_time_model_ridge_regression.pkl  # Production-ready trained model
├── figures/                            # Generated visualizations
│   ├── correlation_heatmap.png
│   ├── regression_diagnostics.png
│   ├── wait_time_distribution.png
│   └── facility_comparison.png
└── requirements.txt                    # Python dependencies
```

## 📚 Analysis Progression

**Week 3: Exploratory Analysis & Baseline Model**
- Data loading and exploration across 139 facility-specialty combinations
- Descriptive statistics and feature engineering
- Initial linear regression model development
- Performance baseline establishment

**Week 4: Model Optimization & Hyperparameter Tuning**
- Advanced feature engineering and selection
- Ridge regression with hyperparameter optimization
- Cross-validation and residual diagnostics
- Final model refinement and production deployment

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
pandas, scikit-learn, matplotlib, seaborn
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd va-healthcare-wait-time-prediction

# Install dependencies
pip install -r requirements.txt

# Launch analysis
jupyter notebook
```

### Viewing the Analysis
1. **Start with**: `W3_Exploratory_Analysis.ipynb` - Understand the data exploration and baseline model
2. **Then review**: `W4_Optimized_Model.ipynb` - See the optimized model and final results
3. **For business context**: Read `EXECUTIVE_SUMMARY.md` first

### Using the Trained Model
```python
import pickle
import pandas as pd
from sklearn.linear_model import Ridge

# Load trained model
with open('va_wait_time_model_ridge_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions on new data
predictions = model.predict(new_facility_data)
```

## 🏆 Key Achievements

### Predictive Accuracy
- **R² Score**: 0.65-0.75 demonstrates strong model predictive power
- **Mean Absolute Error**: 8-12 days provides clinically useful precision
- **RMSE**: 12-18 days suitable for operational planning
- **Model Robustness**: Cross-validation confirms consistent performance

### Operational Insights
- **Established Patient Wait Times**: Strong predictor of new patient waits (β ≈ 0.6-0.8)
- **Facility Variations**: Up to 20 days difference between locations identifies optimization opportunities
- **Specialty Patterns**: Consistent relationships enable targeted resource allocation
- **Statistical Rigor**: Residual analysis confirms linear regression assumptions satisfied

## 📈 Model Performance Validation

### Regression Diagnostics
- **Assumption Testing**: Linearity, homoscedasticity, and normality verified
- **Residual Analysis**: Random distribution confirms model appropriateness
- **Feature Significance**: Facility and specialty variables statistically significant
- **Cross-Validation**: Consistent performance across facility subsets

## 🛠️ Technical Implementation

### Core Technologies
- **Language**: Python 3.8+
- **ML Framework**: Scikit-learn for linear regression modeling
- **Data Processing**: Pandas for data manipulation and analysis
- **Visualization**: Matplotlib and Seaborn for analysis plots
- **Development**: Jupyter Notebook for interactive analysis

### Model Architecture
- **Algorithm**: Linear Regression with Ridge regularization
- **Features**: Facility characteristics, specialty types, historical wait patterns
- **Training Data**: 139 facility-specialty combinations
- **Validation**: K-fold cross-validation with stratified sampling

## � Future Enhancements

### Advanced Analytics
- **Time Series Integration** - Capture seasonal wait time patterns
- **Demand Forecasting** - Predict appointment volume changes
- **Resource Optimization** - Recommend staffing adjustments
- **Automated Alerts** - Flag predicted capacity constraints

### Operational Expansion
- **Facility Dashboard** - Real-time wait time predictions
- **Specialty-Specific Models** - Separate models for different medical specialties
- **Geographic Analysis** - Regional variation analysis
- **Intervention Planning** - Evidence-based scheduling recommendations

## 👤 Author

**Michael Bubulka**
- **Background**: Data Science & Healthcare Analytics
- **LinkedIn**: [linkedin.com/in/michaelbubulka](https://linkedin.com/in/michaelbubulka)
- **Portfolio**: [bubulkaanalytics.com](https://bubulkaanalytics.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This project uses anonymized VA wait time data. All analyses comply with healthcare data protection requirements.

---

*This project demonstrates the practical application of machine learning to improve healthcare operations and veteran care delivery.*