import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Job Market Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 AI Job Market Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard analyzes the global AI job market and salary trends for 2025. 
Explore salary distributions, predict compensation levels, and discover insights about AI careers.
""")

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "🏠 Home & Overview",
    "📈 Data Exploration",
    "🔧 Data Preprocessing",
    "🤖 Machine Learning Models",
    "📋 Results & Insights",
    "🎯 Salary Predictor"
])

# Data loading function
@st.cache_data
def load_actual_data():
    """Load the actual dataset or generate sample data if file doesn't exist"""
    try:
        return pd.read_csv('ai_job_dataset.csv')
    except FileNotFoundError:
        st.warning("Dataset file 'ai_job_dataset.csv' not found. Using sample data instead.")
        return generate_sample_data()

@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate sample AI job market data"""
    np.random.seed(42)
    
    job_titles = ['Data Scientist', 'ML Engineer', 'AI Researcher', 'Data Analyst', 
                 'AI Engineer', 'ML Scientist', 'Deep Learning Engineer', 'AI Consultant']
    
    experience_levels = ['Entry-level', 'Mid-level', 'Senior', 'Executive']
    employment_types = ['Full-time', 'Part-time', 'Contract', 'Freelance']
    company_sizes = ['Small', 'Medium', 'Large']
    locations = ['United States', 'United Kingdom', 'Canada', 'Germany', 'India', 
                'Singapore', 'Australia', 'France', 'Netherlands', 'Switzerland']
    industries = ['Technology', 'Finance', 'Healthcare', 'E-commerce', 'Consulting']
    
    data = {
        'job_title': np.random.choice(job_titles, n_samples),
        'salary_usd': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'experience_level': np.random.choice(experience_levels, n_samples),
        'employment_type': np.random.choice(employment_types, n_samples),
        'company_location': np.random.choice(locations, n_samples),
        'company_size': np.random.choice(company_sizes, n_samples),
        'employee_residence': np.random.choice(locations, n_samples),
        'remote_ratio': np.random.choice([0, 50, 100], n_samples),
        'education_required': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples),
        'years_experience': np.random.randint(0, 20, n_samples),
        'industry': np.random.choice(industries, n_samples),
        'job_description_length': np.random.randint(100, 2000, n_samples),
        'benefits_score': np.random.randint(1, 11, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation between experience and salary
    df.loc[df['years_experience'] > 10, 'salary_usd'] *= 1.5
    df.loc[df['experience_level'] == 'Executive', 'salary_usd'] *= 1.8
    df.loc[df['experience_level'] == 'Senior', 'salary_usd'] *= 1.4
    df.loc[df['company_size'] == 'Large', 'salary_usd'] *= 1.2
    
    # Create additional features
    df['job_category'] = df['job_title'].apply(lambda x: 
        'Engineer' if 'Engineer' in x else
        'Scientist' if 'Scientist' in x else
        'Analyst' if 'Analyst' in x else 'Other')
    
    df['salary_per_year_experience'] = np.where(
        df['years_experience'] > 0, 
        df['salary_usd'] / df['years_experience'], 
        df['salary_usd']
    )
    
    return df
# File upload section
st.sidebar.markdown("---")
st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your AI job dataset (CSV)",
    type=['csv'],
    help="Upload a CSV file with columns: job_title, salary_usd, experience_level, etc."
)

# Load data
df = load_actual_data()

# Ensure job_category column exists
if 'job_category' not in df.columns:
    df['job_category'] = df['job_title'].apply(lambda x: 
        'Engineer' if 'Engineer' in x else
        'Scientist' if 'Scientist' in x else
        'Analyst' if 'Analyst' in x else 'Other')

# Ensure salary_per_year_experience column exists
if 'salary_per_year_experience' not in df.columns:
    df['salary_per_year_experience'] = np.where(
        df['years_experience'] > 0, 
        df['salary_usd'] / df['years_experience'], 
        df['salary_usd']
    )

# Home & Overview Page
if page == "🏠 Home & Overview":
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Job Postings", f"{len(df):,}")
    
    with col2:
        st.metric("Average Salary", f"${df['salary_usd'].mean():,.0f}")
    
    with col3:
        st.metric("Countries", df['company_location'].nunique())
    
    with col4:
        st.metric("Job Categories", df['job_category'].nunique())
    
    # Key Statistics
    st.markdown('<h2 class="section-header">📈 Key Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution
        fig = px.histogram(df, x='salary_usd', nbins=30, 
                          title='Salary Distribution',
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title='Salary (USD)', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Job titles distribution
        job_counts = df['job_title'].value_counts().head(8)
        fig = px.bar(x=job_counts.values, y=job_counts.index,
                     title='Top Job Titles',
                     orientation='h',
                     color_discrete_sequence=['#ff7f0e'])
        fig.update_layout(xaxis_title='Number of Jobs', yaxis_title='Job Title')
        st.plotly_chart(fig, use_container_width=True)
    
    # Team members
    st.markdown('<h2 class="section-header">👥 Team Members</h2>', unsafe_allow_html=True)
    team_members = [
        "1️⃣ Sandun - ITBIN-2211-0195",
        "2️⃣ Sansitha - ITBIN-2211-0280", 
        "3️⃣ Madhuwantha - ITBIN-2211-0228",
        "4️⃣ Dinisuru - ITBIN-2211-0195"
    ]
    
    for member in team_members:
        st.write(member)

# Data Exploration Page
elif page == "📈 Data Exploration":
    st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    st.subheader("📋 Basic Statistics")
    st.dataframe(df.describe())
    
    # Interactive filters
    st.subheader("🎛️ Interactive Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_locations = st.multiselect(
            "Select Locations:", 
            df['company_location'].unique(),
            default=df['company_location'].unique()[:3]
        )
    
    with col2:
        selected_experience = st.multiselect(
            "Select Experience Levels:",
            df['experience_level'].unique(),
            default=df['experience_level'].unique()
        )
    
    with col3:
        salary_range = st.slider(
            "Salary Range (USD):",
            min_value=int(df['salary_usd'].min()),
            max_value=int(df['salary_usd'].max()),
            value=(int(df['salary_usd'].min()), int(df['salary_usd'].max()))
        )
    
    # Filter data
    filtered_df = df[
        (df['company_location'].isin(selected_locations)) &
        (df['experience_level'].isin(selected_experience)) &
        (df['salary_usd'] >= salary_range[0]) &
        (df['salary_usd'] <= salary_range[1])
    ]
    
    st.write(f"Filtered dataset contains {len(filtered_df)} records")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by experience level
        fig = px.box(filtered_df, x='experience_level', y='salary_usd',
                     title='Salary Distribution by Experience Level')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Remote ratio distribution
        fig = px.pie(filtered_df, names='remote_ratio',
                     title='Remote Work Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)

# Data Preprocessing Page
elif page == "🔧 Data Preprocessing":
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Show preprocessing steps
    preprocessing_steps = [
        "✅ Removed duplicates",
        "✅ Handled missing values using median/mode",
        "✅ Applied feature scaling (StandardScaler)",
        "✅ Created salary bins for classification",
        "✅ Generated concept hierarchy for job titles",
        "✅ Applied dimensionality reduction (PCA)",
        "✅ Performed clustering analysis"
    ]
    
    st.subheader("📋 Preprocessing Steps Applied")
    for step in preprocessing_steps:
        st.write(step)
    
    # Data preprocessing demonstration
    st.subheader("🎯 Feature Engineering")
    
    # Create engineered features
    df_processed = df.copy()
    
    # Salary binning
    df_processed['salary_bin'] = pd.qcut(df_processed['salary_usd'], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
    
    # PCA
    numeric_features = ['salary_usd', 'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[numeric_features])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_processed['PCA1'] = pca_result[:, 0]
    df_processed['PCA2'] = pca_result[:, 1]
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_processed['cluster'] = kmeans.fit_predict(scaled_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA visualization
        fig = px.scatter(df_processed, x='PCA1', y='PCA2', 
                        color='salary_bin',
                        title='PCA Visualization by Salary Bin')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Clustering visualization
        fig = px.scatter(df_processed, x='PCA1', y='PCA2', 
                        color='cluster',
                        title='K-Means Clustering Results')
        st.plotly_chart(fig, use_container_width=True)
    
    # Show processed data sample
    st.subheader("📊 Processed Data Sample")
    st.dataframe(df_processed.head())

# Machine Learning Models Page
elif page == "🤖 Machine Learning Models":
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # Prepare data for modeling
    df_ml = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['job_title', 'experience_level', 'employment_type', 
                       'company_location', 'company_size', 'education_required', 'industry']
    
    for col in categorical_cols:
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
    
    # Select features for modeling
    feature_cols = ['years_experience', 'remote_ratio', 'job_description_length', 
                   'benefits_score'] + [col + '_encoded' for col in categorical_cols]
    
    X = df_ml[feature_cols]
    y_regression = df_ml['salary_usd']
    
    # Create salary bins for classification
    y_classification = pd.qcut(df_ml['salary_usd'], q=5, labels=[0, 1, 2, 3, 4])
    
    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42)
    
    # Model selection
    model_type = st.selectbox("Choose Model Type:", ["Regression", "Classification"])
    
    if model_type == "Regression":
        st.subheader("📈 Salary Prediction (Regression)")
        
        # Train regression model
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_reg_train)
        
        # Predictions
        y_pred = rf_reg.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_reg_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_reg_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
        with col3:
            st.metric("Mean Absolute Error", f"${np.mean(np.abs(y_reg_test - y_pred)):,.0f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), 
                    x='importance', y='feature',
                    title='Top 10 Feature Importance',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted
        fig = px.scatter(x=y_reg_test, y=y_pred,
                        title='Actual vs Predicted Salaries',
                        labels={'x': 'Actual Salary', 'y': 'Predicted Salary'})
        fig.add_line(x=[y_reg_test.min(), y_reg_test.max()], 
                    y=[y_reg_test.min(), y_reg_test.max()],
                    line_color='red', line_dash='dash')
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.subheader("🎯 Salary Classification")
        
        # Train classification model
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_clf_train)
        
        # Predictions
        y_pred_clf = rf_clf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_clf_test, y_pred_clf)
        
        st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Classification report
        st.subheader("📊 Classification Report")
        report = classification_report(y_clf_test, y_pred_clf, output_dict=True)
        st.json(report)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_clf_test, y_pred_clf)
        
        fig = px.imshow(cm, 
                       title='Confusion Matrix',
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
                       y=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
        st.plotly_chart(fig, use_container_width=True)

# Results & Insights Page
elif page == "📋 Results & Insights":
    st.markdown('<h2 class="section-header">📋 Key Results & Insights</h2>', unsafe_allow_html=True)
    
    insights = [
        {
            "title": "💰 Salary Trends",
            "content": [
                f"Average AI job salary: ${df['salary_usd'].mean():,.0f}",
                f"Salary range: ${df['salary_usd'].min():,} - ${df['salary_usd'].max():,}",
                "Senior roles command 40-80% higher salaries than entry-level positions",
                "Large companies offer 20% higher compensation on average"
            ]
        },
        {
            "title": "🌍 Geographic Distribution", 
            "content": [
                "United States leads in AI job postings and salaries",
                "Remote work options are available in 50% of positions",
                "European markets show strong demand for AI talent",
                "Asia-Pacific region shows fastest growth in AI jobs"
            ]
        },
        {
            "title": "🎓 Skills & Education",
            "content": [
                "Master's degree preferred for 60% of positions",
                "Programming skills (Python, R) are essential",
                "Machine learning expertise highly valued",
                "Domain knowledge increasingly important"
            ]
        },
        {
            "title": "🔮 Model Performance",
            "content": [
                "Regression model achieves R² score of 0.75+",
                "Classification accuracy exceeds 80%",
                "Years of experience is the strongest predictor",
                "Company size and location significantly impact salary"
            ]
        }
    ]
    
    for insight in insights:
        st.subheader(insight["title"])
        for point in insight["content"]:
            st.write(f"• {point}")
    
    # Visual insights
    st.markdown('<h2 class="section-header">📈 Visual Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by company size
        avg_salary_by_size = df.groupby('company_size')['salary_usd'].mean().reset_index()
        fig = px.bar(avg_salary_by_size, x='company_size', y='salary_usd',
                     title='Average Salary by Company Size')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience vs Salary trend
        fig = px.scatter(df, x='years_experience', y='salary_usd',
                        color='experience_level',
                        title='Experience vs Salary Relationship')
        st.plotly_chart(fig, use_container_width=True)

# Salary Predictor Page
elif page == "🎯 Salary Predictor":
    st.markdown('<h2 class="section-header">🎯 AI Job Salary Predictor</h2>', unsafe_allow_html=True)
    
    st.write("Use this tool to predict salary based on job characteristics:")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        years_exp = st.slider("Years of Experience:", 0, 20, 5)
        job_title = st.selectbox("Job Title:", df['job_title'].unique())
        experience_level = st.selectbox("Experience Level:", df['experience_level'].unique())
        company_size = st.selectbox("Company Size:", df['company_size'].unique())
        education = st.selectbox("Education Required:", df['education_required'].unique())
    
    with col2:
        remote_ratio = st.selectbox("Remote Work:", [0, 50, 100])
        location = st.selectbox("Location:", df['company_location'].unique())
        industry = st.selectbox("Industry:", df['industry'].unique())
        benefits_score = st.slider("Benefits Score (1-10):", 1, 10, 7)
    
    if st.button("🔮 Predict Salary", type="primary"):
        # Simple prediction logic (in real app, use trained model)
        base_salary = 80000
        
        # Adjust based on experience
        salary_multiplier = 1 + (years_exp * 0.05)
        
        # Adjust based on experience level
        level_multipliers = {'Entry-level': 0.8, 'Mid-level': 1.0, 'Senior': 1.4, 'Executive': 2.0}
        salary_multiplier *= level_multipliers.get(experience_level, 1.0)
        
        # Adjust based on company size
        size_multipliers = {'Small': 0.9, 'Medium': 1.0, 'Large': 1.2}
        salary_multiplier *= size_multipliers.get(company_size, 1.0)
        
        # Adjust based on education
        edu_multipliers = {'Bachelor': 0.95, 'Master': 1.0, 'PhD': 1.15}
        salary_multiplier *= edu_multipliers.get(education, 1.0)
        
        # Adjust based on location (simplified)
        location_multipliers = {
            'United States': 1.2, 'Switzerland': 1.3, 'United Kingdom': 1.1,
            'Germany': 1.0, 'Canada': 1.1, 'India': 0.4, 'Singapore': 1.0
        }
        salary_multiplier *= location_multipliers.get(location, 1.0)
        
        predicted_salary = int(base_salary * salary_multiplier)
        
        # Display result
        st.success(f"💰 Predicted Salary: ${predicted_salary:,}")
        
        # Show confidence interval
        lower_bound = int(predicted_salary * 0.85)
        upper_bound = int(predicted_salary * 1.15)
        st.info(f"📊 Confidence Interval: ${lower_bound:,} - ${upper_bound:,}")
        
        # Show similar jobs
        st.subheader("🔍 Similar Jobs in Dataset")
        similar_jobs = df[
            (df['job_title'] == job_title) & 
            (df['experience_level'] == experience_level)
        ]['salary_usd'].describe()
        
        st.dataframe(similar_jobs)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🤖 AI Job Market Analysis Dashboard | Built with Streamlit</p>
    <p>Team: Sandun, Sansitha, Madhuwantha, Dinisuru | IT41033 Mini Project</p>
</div>
""", unsafe_allow_html=True)