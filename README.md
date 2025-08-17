# 🤖 AI Job Market Analysis Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **An interactive data analytics platform for exploring global AI job market trends and salary predictions**

![Dashboard Preview](https://via.placeholder.com/800x400/1f77b4/ffffff?text=AI+Job+Market+Dashboard)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

The **AI Job Market Analysis Dashboard** is a comprehensive data science project that provides deep insights into the artificial intelligence job market for 2025. Built with Streamlit and powered by machine learning, this interactive platform helps job seekers, employers, and researchers understand salary trends, job requirements, and market dynamics in the AI industry.

### Key Objectives:
- 📊 Analyze global AI job market trends
- 💰 Predict salary ranges using machine learning
- 🌍 Explore geographic distribution of opportunities
- 🎓 Understand skill and education requirements
- 📈 Provide actionable insights for career planning

## 🚀 Features

### 📊 **Interactive Data Exploration**
- **Dynamic Filtering**: Filter jobs by location, experience level, salary range
- **Real-time Visualizations**: Interactive charts using Plotly
- **Statistical Analysis**: Comprehensive data summaries and correlations
- **Custom Data Upload**: Support for your own CSV datasets

### 🤖 **Machine Learning Models**
- **Salary Prediction**: Random Forest regression model with 75+ R² score
- **Classification**: Multi-class salary range prediction
- **Feature Importance**: Understand key factors affecting compensation
- **Model Performance Metrics**: Detailed accuracy and error analysis

### 🔧 **Data Preprocessing**
- **Automated Cleaning**: Handle missing values and duplicates
- **Feature Engineering**: Create derived features and categories
- **Dimensionality Reduction**: PCA for data visualization
- **Clustering Analysis**: K-means clustering for job segmentation

### 📈 **Advanced Analytics**
- **Correlation Analysis**: Heatmaps showing feature relationships
- **Trend Analysis**: Salary trends by experience and location
- **Geographic Insights**: Country-wise job distribution
- **Industry Breakdown**: Sector-wise analysis

### 🎯 **Salary Predictor Tool**
- **AI-Powered Predictions**: Estimate salaries based on job characteristics
- **Confidence Intervals**: Statistical confidence ranges
- **Similar Jobs Analysis**: Compare with historical data
- **Interactive Input Form**: Easy-to-use prediction interface

## 🎬 Demo

### Live Dashboard
🌐 **[View Live Demo](your-streamlit-app-url.com)** *(Replace with actual URL)*

### Screenshots

<details>
<summary>📊 Dashboard Screenshots</summary>

| Home Overview | Data Exploration | ML Models |
|---------------|-----------------|-----------|
| ![Home](https://via.placeholder.com/250x150/1f77b4/ffffff?text=Home) | ![Exploration](https://via.placeholder.com/250x150/ff7f0e/ffffff?text=Data+Explorer) | ![ML](https://via.placeholder.com/250x150/2ca02c/ffffff?text=ML+Models) |

| Salary Predictor | About | Data Upload |
|-----------------|-------|-------------|
| ![Predictor](https://via.placeholder.com/250x150/d62728/ffffff?text=Predictor) | ![About](https://via.placeholder.com/250x150/9467bd/ffffff?text=About) | ![Upload](https://via.placeholder.com/250x150/8c564b/ffffff?text=Upload) |

</details>

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/ai-job-market-dashboard.git
cd ai-job-market-dashboard
```

### Step 2: Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual Installation**
```bash
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn openpyxl
```

**Option C: Using Virtual Environment (Best Practice)**
```bash
# Create virtual environment
python -m venv dashboard_env

# Activate environment
# Windows:
dashboard_env\Scripts\activate
# macOS/Linux:
source dashboard_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📖 Usage

### 1. **Getting Started**
- Launch the dashboard using `streamlit run app.py`
- Navigate using the sidebar menu
- Upload your own CSV data or use sample data

### 2. **Data Upload**
- Use the sidebar file uploader to import your CSV
- Supported columns: `job_title`, `salary_usd`, `experience_level`, etc.
- Missing columns will be auto-generated with defaults

### 3. **Exploration**
- **Home & Overview**: View key statistics and distributions
- **Data Exploration**: Apply filters and create custom visualizations
- **Data Preprocessing**: Understand data cleaning and feature engineering

### 4. **Machine Learning**
- **ML Models**: Train and evaluate regression/classification models
- **Results & Insights**: Review key findings and recommendations
- **Salary Predictor**: Get personalized salary estimates

### 5. **Navigation Guide**
```
🏠 Home & Overview     → Key statistics and team info
📈 Data Exploration    → Interactive filtering and visualization
🔧 Data Preprocessing  → Feature engineering and PCA
🤖 ML Models          → Train and evaluate models
📋 Results & Insights  → Key findings and recommendations
🎯 Salary Predictor    → AI-powered salary estimation
ℹ️ About              → Project documentation and help
```

## 📊 Dataset

### Expected CSV Format
```csv
job_title,salary_usd,experience_level,company_location,years_experience,company_size,remote_ratio,education_required,industry,employment_type,job_description_length,benefits_score
Data Scientist,95000,Senior,United States,7,Large,100,Master,Technology,Full-time,1500,8
ML Engineer,110000,Senior,United States,5,Medium,50,Master,Technology,Full-time,1200,9
```

### Required Columns
- `job_title`: Job position name
- `salary_usd`: Annual salary in USD
- `experience_level`: Entry-level, Mid-level, Senior, Executive
- `company_location`: Country or region
- `years_experience`: Years of relevant experience

### Optional Columns
- `company_size`: Small, Medium, Large
- `remote_ratio`: 0, 50, 100 (percentage remote)
- `education_required`: Bachelor, Master, PhD
- `industry`: Technology, Finance, Healthcare, etc.
- `employment_type`: Full-time, Part-time, Contract
- `benefits_score`: Rating from 1-10

### Sample Data
If no CSV is uploaded, the dashboard generates realistic sample data with 1000+ records for demonstration.

## 🛠 Technology Stack

### **Frontend & Visualization**
- **Streamlit**: Interactive web framework
- **Plotly Express**: Dynamic, interactive charts
- **HTML/CSS**: Custom styling and layout
- **Matplotlib/Seaborn**: Statistical visualizations

### **Data Processing**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and arrays
- **StandardScaler**: Feature normalization
- **LabelEncoder**: Categorical variable encoding

### **Machine Learning**
- **Scikit-learn**: ML algorithms and metrics
- **Random Forest**: Regression and classification
- **PCA**: Dimensionality reduction
- **K-Means**: Clustering analysis

### **Development Tools**
- **Python 3.8+**: Core programming language
- **Git**: Version control
- **VS Code**: Development environment
- **Jupyter**: Data exploration and prototyping

## 📁 Project Structure

```
ai-job-market-dashboard/
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📊 ai_job_dataset.csv        # Sample dataset (optional)
├── 📁 assets/                   # Images and static files
│   ├── 🖼️ dashboard_preview.png
│   └── 🖼️ screenshots/
├── 📁 docs/                     # Additional documentation
│   ├── 📄 api_reference.md
│   └── 📄 user_guide.md
├── 📁 notebooks/               # Jupyter notebooks for analysis
│   ├── 📓 data_exploration.ipynb
│   └── 📓 model_development.ipynb
└── 📁 tests/                   # Unit tests
    ├── 📄 test_data_processing.py
    └── 📄 test_models.py
```

## 👥 Team

This project was developed as part of **IT41033 - Data Science and Analytics Mini Project** by:

| Team Member | Student ID | Role | Contributions |
|-------------|------------|------|---------------|
| **Sandun** | ITBIN-2211-0195 | 🎯 Project Lead & Data Analysis | Data preprocessing, statistical analysis, project coordination |
| **Sansitha** | ITBIN-2211-0280 | 🤖 Machine Learning & Modeling | ML model development, feature engineering, model evaluation |
| **Madhuwantha** | ITBIN-2211-0228 | 🎨 Frontend & Visualization | Streamlit interface, Plotly charts, UI/UX design |
| **Dinisuru** | ITBIN-2211-0195 | 🔧 Data Engineering & Testing | Data pipeline, testing, deployment, documentation |

### Course Information
- **Course**: IT41033 - Data Science and Analytics
- **Institution**: [Your Institution Name]
- **Academic Year**: 2025
- **Project Duration**: [Start Date] - [End Date]

## 🤝 Contributing

We welcome contributions to improve the dashboard! Here's how you can help:

### For Students & Researchers
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas
- 📊 **New visualizations** or chart types
- 🤖 **Additional ML models** or algorithms  
- 🎨 **UI/UX improvements** and styling
- 📚 **Documentation** and tutorials
- 🐛 **Bug fixes** and performance optimization
- 📈 **New features** and functionality

### Development Setup
1. Clone the repository
2. Create virtual environment: `python -m venv dev_env`
3. Activate environment: `source dev_env/bin/activate` (Linux/Mac) or `dev_env\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Install development dependencies: `pip install pytest black flake8`
6. Make your changes and test
7. Submit pull request

## 📞 Support & Help

### 🆘 Getting Help
- **📖 Documentation**: Check the built-in About section
- **🐛 Issues**: Report bugs via GitHub Issues
- **💡 Questions**: Use GitHub Discussions
- **📧 Contact**: [team-email@institution.edu]

### 🔧 Troubleshooting

<details>
<summary>Common Issues & Solutions</summary>

**Problem**: `ModuleNotFoundError: No module named 'plotly'`
```bash
pip install plotly
```

**Problem**: Dashboard won't start
```bash
# Check Python version
python --version

# Upgrade Streamlit
pip install --upgrade streamlit

# Run with verbose output
streamlit run app.py --server.headless true
```

**Problem**: Charts not displaying
- Clear browser cache
- Try different browser
- Check console for JavaScript errors

**Problem**: CSV upload fails
- Ensure CSV has required columns
- Check file encoding (UTF-8 recommended)
- Verify data types match expected format

</details>

## 📊 Performance Metrics

### Model Performance
- **Regression Model**: R² Score of 0.75+, RMSE ~$15,000-20,000
- **Classification Model**: 80%+ accuracy across 5 salary bins
- **Prediction Speed**: <100ms for salary estimation
- **Data Processing**: Handles datasets up to 100,000+ records

### Dashboard Performance
- **Load Time**: <5 seconds initial load
- **Interactivity**: Real-time filtering and visualization
- **Memory Usage**: Optimized with Streamlit caching
- **Browser Support**: Chrome, Firefox, Safari, Edge

## 🚀 Future Enhancements

### Planned Features
- [ ] 🌐 **Multi-language Support**: Dashboard in multiple languages
- [ ] 📱 **Mobile Optimization**: Responsive design for mobile devices
- [ ] 🔄 **Real-time Data**: Integration with job market APIs
- [ ] 🧠 **Advanced ML**: Deep learning models for better predictions
- [ ] 📈 **Time Series**: Historical trend analysis and forecasting
- [ ] 🎯 **Job Matching**: Recommend jobs based on user profile
- [ ] 📊 **Custom Reports**: Export detailed PDF reports
- [ ] 🔐 **User Authentication**: Personal dashboards and saved preferences

### Research Extensions
- **Geographic Analysis**: City-level granularity
- **Skills Analysis**: NLP on job descriptions
- **Market Trends**: Economic indicator correlations
- **Industry Insights**: Sector-specific deep dives

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 AI Job Market Dashboard Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **📚 Course Instructors**: For guidance and support throughout the project
- **🎓 Institution**: For providing resources and learning environment  
- **🌟 Streamlit Community**: For excellent documentation and examples
- **📊 Plotly Team**: For powerful visualization capabilities
- **🤖 Scikit-learn Contributors**: For comprehensive ML tools
- **💡 Open Source Community**: For inspiration and code examples

## 📊 Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/ai-job-market-dashboard)
![GitHub language count](https://img.shields.io/github/languages/count/yourusername/ai-job-market-dashboard)
![GitHub top language](https://img.shields.io/github/languages/top/yourusername/ai-job-market-dashboard)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/ai-job-market-dashboard)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-job-market-dashboard?style=social)](https://github.com/yourusername/ai-job-market-dashboard)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-job-market-dashboard?style=social)](https://github.com/yourusername/ai-job-market-dashboard/fork)

**Built with ❤️ by the AI Job Market Dashboard Team**

[🔝 Back to top](#-ai-job-market-analysis-dashboard)

</div>
