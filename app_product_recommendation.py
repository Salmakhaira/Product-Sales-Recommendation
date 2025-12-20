import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import json
from pathlib import Path
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go
import plotly.express as px

# Keras/TF
try:
    from keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Product Sales Recommendation System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling (keep fancy for UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .recommendation-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        animation: slideUp 0.5s ease-out;
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .layak {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .cukup-layak {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .tidak-layak {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .info-card {
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
    }
    
    .info-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        transform: translateX(5px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        flex: 1;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 2rem 1rem;
    }
    
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
    }
    
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .product-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .history-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid white;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #38ef7d;
        color: white;
    }
    
    .badge-warning {
        background: #f5576c;
        color: white;
    }
    
    .badge-info {
        background: #00f2fe;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state init
if 'history' not in st.session_state:
    st.session_state.history = []
if 'product_stats' not in st.session_state:
    st.session_state.product_stats = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

@st.cache_resource
def load_models():
    """Load sentiment model, XGBoost, vectorizer, and config"""
    try:
        required_files = [
            'best_sentiment_model.h5',
            'xgboost_recommendation_model.pkl',
            'tfidf_vectorizer.pkl',
            'model_config.json'
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            st.error(f"Missing files: {', '.join(missing)}")
            st.info("Run the training notebook first")
            return None, None, None, None
        
        sentiment_model = load_model('best_sentiment_model.h5')
        xgb_model = joblib.load('xgboost_recommendation_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return sentiment_model, xgb_model, vectorizer, config
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

@st.cache_data
def load_dataset(file_path):
    """Load CSV dataset"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_text(text, use_stemming=False, use_stopword=True):
    """Clean Indonesian text - remove URLs, special chars, apply stopword removal"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    # Remove URLs, HTML, emails, mentions, hashtags, numbers
    text = re.sub(r'http\S+|www\S+|<.*?>|\S+@\S+|@\w+|#\w+|\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > 0 and use_stopword:
        try:
            stopword_factory = StopWordRemoverFactory()
            stopword_remover = stopword_factory.create_stop_word_remover()
            text = stopword_remover.remove(text)
        except:
            pass
    
    return text

def get_valid_products(df, config):
    """Filter products that have valid reviews after cleaning"""
    valid = []
    
    for product in df[config['columns']['product']].unique():
        subset = df[df[config['columns']['product']] == product].copy()
        subset['cleaned'] = subset[config['columns']['review']].apply(
            lambda x: preprocess_text(x, use_stemming=False)
        )
        if (subset['cleaned'].str.len() > 0).any():
            valid.append(product)
    
    return valid

def predict_product_recommendation(product_name, df, sentiment_model, vectorizer, 
                                  xgb_model, config):
    """Main prediction function - combines DNN sentiment + XGBoost recommendation"""
    
    PRODUCT_COLUMN = config['columns']['product']
    REVIEW_COLUMN = config['columns']['review']
    RATING_COLUMN = config['columns']['rating']
    
    # Get product reviews
    product_reviews = df[df[PRODUCT_COLUMN] == product_name].copy()
    
    if len(product_reviews) == 0:
        return {
            'error': f"Product '{product_name}' not found",
            'available_products': df[PRODUCT_COLUMN].unique()[:10].tolist()
        }
    
    # Clean reviews
    product_reviews['cleaned_review'] = product_reviews[REVIEW_COLUMN].apply(
        lambda x: preprocess_text(x, use_stemming=False)
    )
    
    product_reviews = product_reviews[product_reviews['cleaned_review'].str.len() > 0]
    
    if len(product_reviews) == 0:
        return {'error': 'No valid reviews found'}
    
    review_count = len(product_reviews)
    
    # DNN sentiment prediction (raw)
    reviews_tfidf = vectorizer.transform(product_reviews['cleaned_review'])
    sentiment_raw = sentiment_model.predict(reviews_tfidf.toarray()).flatten()
    
    # Rating-guided adjustment
    def adjust_sentiment(rating, raw_sent):
        if rating <= 1:
            return min(raw_sent, 0.35)
        elif rating <= 2:
            return min(raw_sent, 0.50)
        elif rating <= 3:
            return max(min(raw_sent, 0.70), 0.40)
        elif rating <= 4:
            return max(min(raw_sent, 0.90), 0.65)
        else:
            return min(max(raw_sent, 0.80), 0.95)
    
    product_reviews['sentiment_score'] = [
        adjust_sentiment(r, s) for r, s in 
        zip(product_reviews[RATING_COLUMN], sentiment_raw)
    ]
    
    # Calculate features
    features = {
        'avg_sentiment': product_reviews['sentiment_score'].mean(),
        'sentiment_std': product_reviews['sentiment_score'].std(),
        'positive_rate': (product_reviews[RATING_COLUMN] >= 4).mean(),
        'avg_rating': product_reviews[RATING_COLUMN].mean(),
        'rating_std': product_reviews[RATING_COLUMN].std(),
        'review_count': review_count
    }
    
    # Handle edge cases (single review or identical values)
    if pd.isna(features['sentiment_std']) or features['sentiment_std'] == 0:
        features['sentiment_std'] = 0.1
    if pd.isna(features['rating_std']) or features['rating_std'] == 0:
        features['rating_std'] = 0.1
    
    # XGBoost prediction
    feature_names = ['avg_sentiment', 'sentiment_std', 'positive_rate', 
                    'avg_rating', 'rating_std', 'review_count']
    X_pred = np.array([[features[f] for f in feature_names]])
    
    raw_pred = float(xgb_model.predict(X_pred)[0])
    score = np.clip(raw_pred, 0, 1)
    
    # Categorize
    if score >= 0.75:
        category = "Layak Dijual"
        desc = "This product type is recommended for sale due to its excellent performance!"
        emoji = "✅"
        box_class = "layak"
    elif score >= 0.50:
        category = "Cukup Layak Dijual"
        desc = "This product is reasonably suitable for sale with moderate performance."
        emoji = "⚠️"
        box_class = "cukup-layak"
    else:
        category = "Tidak Layak Dijual"
        desc = "This product is not recommended for sale based on current reviews and ratings."
        emoji = "❌"
        box_class = "tidak-layak"
    
    # Sample reviews
    samples = product_reviews.sample(min(5, len(product_reviews))).to_dict('records')
    
    return {
        'recommendation_score': score,
        'recommendation_category': category,
        'recommendation_emoji': emoji,
        'description': desc,
        'box_class': box_class,
        'statistics': {
            'positive_rate': features['positive_rate'] * 100,
            'avg_rating': features['avg_rating'],
            'review_count': features['review_count'],
            'avg_sentiment': features['avg_sentiment'],
            'sentiment_std': features['sentiment_std']
        },
        'sample_reviews': samples,
        'debug_info': {
            'raw_prediction': raw_pred,
            'model_type': 'XGBRegressor',
            'features': features
        }
    }

def main():
    # Hero section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">📦 Product Sales Recommendation System</div>
            <div class="hero-subtitle">AI-Powered Product Analysis with Deep Neural Network & XGBoost</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        sentiment_model, xgb_model, vectorizer, config = load_models()
    
    if sentiment_model is None:
        st.error("Failed to load models. Check model files.")
        return
    
    # Sidebar - How to use
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h1 style='color: white; font-size: 2rem;'>📖 How to Use</h1>
            </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""            
            <div style='margin-bottom: 1.2rem;'>
                <strong style='color: #FFD700;'>1️⃣ Upload Dataset</strong>
                <p style='font-size: 0.85rem; margin: 0.3rem 0 0 1rem; line-height: 1.5;'>
                • Go to Product Analysis tab<br>
                • Upload CSV file with Product, Review, Rating columns
                </p>
            </div>
            
            <div style='margin-bottom: 1.2rem;'>
                <strong style='color: #FFD700;'>2️⃣ Auto-Generate Statistics</strong>
                <p style='font-size: 0.85rem; margin: 0.3rem 0 0 1rem; line-height: 1.5;'>
                • Statistics automatically generated after upload<br>
                • Processing all products in dataset<br>
                • View results in Statistics Dashboard tab
                </p>
            </div>
            
            <div style='margin-bottom: 1.2rem;'>
                <strong style='color: #FFD700;'>3️⃣ Analyze Product</strong>
                <p style='font-size: 0.85rem; margin: 0.3rem 0 0 1rem; line-height: 1.5;'>
                • Select product from dropdown<br>
                • Click Analyze Product button<br>
                • View detailed recommendation
                </p>
            </div>
            
            <div style='margin-bottom: 1.2rem;'>
                <strong style='color: #FFD700;'>4️⃣ Results</strong>
                <p style='font-size: 0.85rem; margin: 0.3rem 0 0 1rem; line-height: 1.5;'>
                ✅ Layak Dijual (≥75%)<br>
                ⚠️ Cukup Layak Dijual (50-74%)<br>
                ❌ Tidak Layak Dijual (<50%)
                </p>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin-top: 1rem;'>
                <strong style='color: #FFD700;'>💡 Tips:</strong>
                <p style='font-size: 0.8rem; margin: 0.3rem 0 0 0; line-height: 1.4;'>
                • Use Sample mode for large datasets<br>
                • Products with &lt;20 reviews use direct formula<br>
                • Check About System tab for details
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Show stats if available
        if st.session_state.product_stats is not None:
            product_stats = st.session_state.product_stats
            st.markdown("---")
            st.markdown("""
                <div style='color: white; padding: 1rem;'>
                    <h3>📈 Dataset Statistics</h3>
                </div>
            """, unsafe_allow_html=True)
            
            layak = len(product_stats[product_stats['recommendation_category'] == 'Layak Dijual'])
            cukup = len(product_stats[product_stats['recommendation_category'] == 'Cukup Layak Dijual'])
            tidak = len(product_stats[product_stats['recommendation_category'] == 'Tidak Layak Dijual'])
            
            st.markdown(f"""
                <div class='history-item'>
                    <strong>✅ Layak:</strong> {layak}
                </div>
                <div class='history-item'>
                    <strong>⚠️ Cukup Layak:</strong> {cukup}
                </div>
                <div class='history-item'>
                    <strong>❌ Tidak Layak:</strong> {tidak}
                </div>
            """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Product Analysis", "📊 Statistics Dashboard", "ℹ️ About System"])
    
    with tab1:
        st.markdown("### 📁 Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing product reviews",
            type=['csv'],
            help="Upload your e-commerce product reviews dataset"
        )
        
        if uploaded_file is not None:
            df = load_dataset(uploaded_file)
            
            if df is not None:
                st.success(f"Dataset loaded: **{len(df):,}** reviews from **{df[config['columns']['product']].nunique():,}** products")
                
                with st.expander("Preview Dataset", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Auto-generate stats
                if st.session_state.last_uploaded_file != uploaded_file.name:
                    st.info("Generating statistics for all products...")
    
                    total_products = len(df[config['columns']['product']].unique())
                    if total_products > 100:
                        st.warning(f"Processing {total_products:,} products. This may take several minutes...")
    
                    with st.spinner("AI analyzing products..."):
                        products = df[config['columns']['product']].unique()
        
                        all_stats = []
                        progress_bar = st.progress(0)
        
                        for i, product in enumerate(products):
                            try:
                                result = predict_product_recommendation(
                                    product, df, sentiment_model, 
                                    vectorizer, xgb_model, config
                                )
                                if 'error' not in result:
                                    all_stats.append({
                                        config['columns']['product']: product,
                                        'recommendation_score': result['recommendation_score'],
                                        'recommendation_category': result['recommendation_category'],
                                        'avg_rating': result['statistics']['avg_rating'],
                                        'review_count': result['statistics']['review_count'],
                                        'positive_rate': result['statistics']['positive_rate']
                                    })
                            except Exception as e:
                                st.warning(f"Error processing {product[:30]}...: {str(e)}")
        
                            progress_bar.progress((i + 1) / len(products))
        
                        progress_bar.empty()
        
                        if all_stats:
                            st.session_state.product_stats = pd.DataFrame(all_stats)
                            st.session_state.last_uploaded_file = uploaded_file.name
                            st.balloons()
                            st.success(f"Statistics generated for **{len(all_stats):,}** products! Check **Statistics Dashboard** tab.")
                        else:
                            st.error("Failed to generate statistics. Check your data.")
                
                st.markdown("---")
                
                # Product selection
                st.markdown("### 🎯 Select Product to Analyze")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    valid_products = get_valid_products(df, config)
                    products = sorted(valid_products)
                    st.caption(f"📦 {len(products)} products available (from {df[config['columns']['product']].nunique()} total)")
                    selected_product = st.selectbox("Choose a product:", products)
                
                with col2:
                    st.markdown("<div style='margin-top: 52px;'></div>", unsafe_allow_html=True)
                    analyze_button = st.button("🔍 Analyze Product", type="primary", use_container_width=True)
                
                if analyze_button:
                    with st.spinner(f"AI analyzing '{selected_product}'..."):
                        result = predict_product_recommendation(
                            selected_product, df, sentiment_model, 
                            vectorizer, xgb_model, config
                        )
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.markdown("---")
                        st.markdown("### 🎯 Analysis Result")
                        
                        st.markdown(
                            f"""
                            <div class="recommendation-box {result['box_class']}">
                                <div>{result['recommendation_emoji']} {result['recommendation_category']}</div>
                                <div style="font-size: 1.3rem; margin-top: 1rem; opacity: 0.9;">
                                    Confidence Score: {result['recommendation_score']:.1%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.info(f"**Insight:** {result['description']}")
                        
                        # Metrics
                        st.markdown("### 📊 Detailed Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div style="font-size: 2.2rem; color: #667eea; font-weight: 700;">
                                        {result['statistics']['positive_rate']:.1f}%
                                    </div>
                                    <div style="color: #666; font-weight: 600;">Positive Rate</div>
                                    <div style="height: 2rem;"></div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            stars = "⭐" * int(result['statistics']['avg_rating'])
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div style="font-size: 2.5rem; color: #667eea; font-weight: 700;">
                                        {result['statistics']['avg_rating']:.2f}
                                    </div>
                                    <div style="color: #666; font-weight: 500;">Average Rating</div>
                                    <div style="margin-top: 0.00001rem;">{stars}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div style="font-size: 3.2rem; color: #667eea; font-weight: 700;">
                                        {result['statistics']['review_count']:,}
                                    </div>
                                    <div style="color: #666; font-weight: 400;">Total Reviews</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div style="font-size: 3.2rem; color: #667eea; font-weight: 650;">
                                        {result['statistics']['avg_sentiment']:.2f}
                                    </div>
                                    <div style="color: #666; font-weight: 450;">Avg Sentiment</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Visualizations
                        st.markdown("### 📈 Visual Analytics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gauge chart
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = result['recommendation_score'] * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Recommendation Score", 'font': {'size': 24, 'color': '#333'}},
                                delta = {'reference': 50, 'increasing': {'color': "#11998e"}},
                                gauge = {
                                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
                                    'bar': {'color': "#667eea", 'thickness': 0.75},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "#ccc",
                                    'steps': [
                                        {'range': [0, 50], 'color': 'rgba(255, 99, 132, 0.2)'},
                                        {'range': [50, 75], 'color': 'rgba(255, 206, 86, 0.2)'},
                                        {'range': [75, 100], 'color': 'rgba(75, 192, 192, 0.2)'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 75
                                    }
                                }
                            ))
                            
                            fig.update_layout(
                                height=350,
                                paper_bgcolor="white",
                                font={'family': "Poppins", 'size': 14}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Sentiment pie
                            sentiment_data = pd.DataFrame({
                                'Category': ['Positive', 'Neutral', 'Negative'],
                                'Percentage': [
                                    result['statistics']['positive_rate'],
                                    100 - result['statistics']['positive_rate'] - (100 - result['statistics']['positive_rate'])/2,
                                    (100 - result['statistics']['positive_rate'])/2
                                ]
                            })
                            
                            fig2 = px.pie(
                                sentiment_data,
                                values='Percentage',
                                names='Category',
                                title='Sentiment Distribution',
                                color_discrete_sequence=['#11998e', '#f5576c', '#667eea'],
                                hole=0.4
                            )
                            
                            fig2.update_layout(
                                height=350,
                                paper_bgcolor="white",
                                font={'family': "Poppins"}
                            )
                            fig2.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Sample reviews
                        st.markdown("### 📝 Sample Customer Reviews")
                        with st.expander("View Sample Reviews", expanded=True):
                            for i, review in enumerate(result['sample_reviews'], 1):
                                rating = int(review[config['columns']['rating']])
                                stars = "⭐" * rating
                                st.markdown(
                                    f"""
                                    <div class="product-card">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                            <strong style="color: #667eea;">Review #{i}</strong>
                                            <span style="color: #f39c12;">{stars}</span>
                                        </div>
                                        <p style="color: #666; line-height: 1.6;">{review[config['columns']['review']][:300]}...</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                        # Add to history
                        st.session_state.history.append({
                            'product': selected_product,
                            'category': result['recommendation_category'],
                            'score': result['recommendation_score'],
                            'emoji': result['recommendation_emoji']
                        })
        else:
            st.info("Please upload your dataset in the **Product Analysis** tab. Statistics will be automatically generated.")
    
    with tab2:
        st.markdown("### 📊 Product Statistics Overview")
        
        if st.session_state.product_stats is not None:
            product_stats = st.session_state.product_stats
            st.success(f"Statistics available for **{len(product_stats):,}** products")
            
            # Summary cards
            col1, col2, col3 = st.columns(3)
            
            layak = len(product_stats[product_stats['recommendation_category'] == 'Layak Dijual'])
            cukup = len(product_stats[product_stats['recommendation_category'] == 'Cukup Layak Dijual'])
            tidak = len(product_stats[product_stats['recommendation_category'] == 'Tidak Layak Dijual'])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-number">{layak}</div>
                        <div class="stat-label">✅ Layak Dijual</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-number">{cukup}</div>
                        <div class="stat-label">⚠️ Cukup Layak</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-number">{tidak}</div>
                        <div class="stat-label">❌ Tidak Layak</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Category Distribution")
                cat_counts = product_stats['recommendation_category'].value_counts()
                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    color_discrete_sequence=['#11998e', '#f5576c', '#667eea'],
                    hole=0.3
                )
                fig.update_layout(height=400, paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Score Distribution")
                fig2 = px.histogram(
                    product_stats,
                    x='recommendation_score',
                    nbins=30,
                    color_discrete_sequence=['#667eea']
                )
                fig2.update_layout(
                    height=400,
                    paper_bgcolor="white",
                    xaxis_title="Recommendation Score",
                    yaxis_title="Number of Products"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Top products
            st.markdown("### 🏆 Top Recommended Products")
            top_10 = product_stats.nlargest(10, 'recommendation_score')[[
                config['columns']['product'], 'recommendation_score', 
                'recommendation_category', 'avg_rating', 'review_count'
            ]].copy()
            
            top_10['recommendation_score'] = top_10['recommendation_score'].apply(lambda x: f"{x:.3f}")
            top_10['avg_rating'] = top_10['avg_rating'].apply(lambda x: f"{x:.2f} ⭐")
            
            st.dataframe(top_10, use_container_width=True, hide_index=True)
            
        else:
            st.info("Please upload your dataset in the **Product Analysis** tab to see statistics.")
    
    with tab3:
        st.markdown("### ℹ️ About This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="info-card">
                    <h3>🎯 Purpose</h3>
                    <p>This intelligent system analyzes e-commerce products to determine their sales potential based on:</p>
                    <ul>
                        <li>Customer review sentiment analysis</li>
                        <li>Product rating patterns</li>
                        <li>Review volume and engagement</li>
                        <li>Statistical metrics</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="info-card">
                    <h3>🤖 Technology Stack</h3>
                    <p><strong>1. Deep Neural Network (DNN)</strong></p>
                    <ul>
                        <li>Advanced sentiment classification</li>
                        <li>Trained on Indonesian e-commerce data</li>
                        <li>Accuracy: <strong>96.42%</strong></li>
                        <li>Best performing model</li>
                    </ul>
                    <p><strong>2. XGBoost Predictor</strong></p>
                    <ul>
                        <li>Gradient boosting for recommendations</li>
                        <li>Multi-feature analysis</li>
                        <li>High precision predictions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="info-card">
                    <h3>📊 Output Categories</h3>
                    <div style="margin: 1rem 0;">
                        <span class="badge badge-success">✅ Layak Dijual (≥0.75)</span>
                        <p style="margin-left: 1rem; color: #666;">Highly recommended for sale with excellent performance</p>
                    </div>
                    <div style="margin: 1rem 0;">
                        <span class="badge badge-warning">⚠️ Cukup Layak (0.50-0.74)</span>
                        <p style="margin-left: 1rem; color: #666;">Moderately recommended with room for improvement</p>
                    </div>
                    <div style="margin: 1rem 0;">
                        <span class="badge badge-info">❌ Tidak Layak (<0.50)</span>
                        <p style="margin-left: 1rem; color: #666;">Not recommended based on current performance</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="info-card">
                    <h3>🔬 Model Performance</h3>
                    <p>The DNN model was selected after rigorous comparison with multiple algorithms including:</p>
                    <ul>
                        <li>Logistic Regression</li>
                        <li>Random Forest</li>
                        <li>Support Vector Machine (SVM)</li>
                    </ul>
                    <p><strong>Result:</strong> DNN achieved the highest accuracy and F1-score!</p>
                </div>
            """, unsafe_allow_html=True)
    
    # History sidebar
    if st.session_state.history:
        with st.sidebar:
            st.markdown("---")
            st.markdown("""
                <div style='color: white; padding: 1rem;'>
                    <h3>📜 Recent Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for item in reversed(st.session_state.history[-5:]):
                st.markdown(
                    f"""
                    <div class='history-item'>
                        <strong>{item['emoji']} {item['product'][:25]}...</strong><br>
                        <small>{item['category']} ({item['score']:.2f})</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📦 Product Sales Recommendation System</h3>
            <p style="color: #666;">Powered by Deep Neural Network & XGBoost</p>
            <p style="color: #999; font-size: 0.9rem;">Designed for Indonesian E-Commerce Product Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()