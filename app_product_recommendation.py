"""
Product Sales Recommendation System
AI-Powered Product Analysis with Deep Neural Network & XGBoost

Author: Salma
BINUS University - Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from keras.models import load_model
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODERN CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }
    
    /* Sidebar expander styling */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 0.5rem;
        border-left: 3px solid #667eea;
        font-weight: 600;
        color: #667eea;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    /* Custom gradient header */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .gradient-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .gradient-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: transparent;
        border: none;
        color: #666;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }
    
    /* Upload box */
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
        margin: 2rem 0;
    }
    
    /* Recommendation cards */
    .rec-card {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .rec-layak {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .rec-cukup {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .rec-tidak {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    .metric-box h3 {
        margin: 0;
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .metric-box p {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Clean spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'products' not in st.session_state:
    st.session_state.products = None

# ==========================================
# CONFIGURATION
# ==========================================
PRODUCT_COLUMN = 'Nama Produk'
REVIEW_COLUMN = 'comment-'
RATING_COLUMN = 'rating'

# ==========================================
# FUNCTIONS
# ==========================================

@st.cache_resource
def load_text_processors():
    """Load Sastrawi processors"""
    stemmer_factory = StemmerFactory()
    stopword_factory = StopWordRemoverFactory()
    return stemmer_factory.create_stemmer(), stopword_factory.create_stop_word_remover()

def preprocess_text(text, use_stemming=False, use_stopword=True):
    """Preprocess Indonesian text"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if use_stopword:
        stemmer, stopword_remover = load_text_processors()
        text = stopword_remover.remove(text)
    if use_stemming:
        stemmer, stopword_remover = load_text_processors()
        text = stemmer.stem(text)
    return text

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        sentiment_model = load_model('best_sentiment_model.h5')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sales_model = joblib.load('xgboost_sales_classifier.pkl')
        return sentiment_model, vectorizer, sales_model, None
    except Exception as e:
        return None, None, None, str(e)

def predict_recommendation(product_name, df, sentiment_model, vectorizer, sales_model):
    """Predict product recommendation"""
    
    product_reviews = df[df[PRODUCT_COLUMN] == product_name].copy()
    
    if len(product_reviews) == 0:
        return {'error': f"Produk '{product_name}' tidak ditemukan"}
    
    # Sentiment Analysis
    reviews_tfidf = vectorizer.transform(product_reviews['cleaned_review'])
    sentiment_raw = sentiment_model.predict(reviews_tfidf.toarray(), verbose=0).flatten()
    
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
    avg_sentiment = float(product_reviews['sentiment_score'].mean())
    std_sentiment = float(product_reviews['sentiment_score'].std())
    std_sentiment = 0.1 if pd.isna(std_sentiment) or std_sentiment == 0 else std_sentiment
    
    positive_rate = float((product_reviews[RATING_COLUMN] >= 4).mean())
    avg_rating = float(product_reviews[RATING_COLUMN].mean())
    std_rating = float(product_reviews[RATING_COLUMN].std())
    std_rating = 0.1 if pd.isna(std_rating) or std_rating == 0 else std_rating
    
    review_count = len(product_reviews)
    
    # Sales Prediction
    features = pd.DataFrame([{
        'avg_sentiment_score': avg_sentiment,
        'std_sentiment_score': std_sentiment,
        'positive_rate': positive_rate,
        'avg_rating': avg_rating,
        'std_rating': std_rating,
        'review_count': review_count
    }])
    
    sales_proba = float(sales_model.predict_proba(features)[0, 1])
    
    # Hybrid Score
    sentiment_comp = avg_sentiment * 0.4
    sales_comp = sales_proba * 0.3
    positive_comp = positive_rate * 0.2
    review_comp = (min(review_count, 100) / 100.0) * 0.1
    
    score = min(sentiment_comp + sales_comp + positive_comp + review_comp, 1.0)
    
    # Categorize
    if score >= 0.75:
        category = 'Produk Unggulan'
        emoji = '✅'
        css_class = 'rec-layak'
        recommendation = 'PERBANYAK STOK'
        action_detail = 'Tingkatkan stok 30-50%'
    elif score >= 0.5:
        category = 'Produk Stabil'
        emoji = '⚠️'
        css_class = 'rec-cukup'
        recommendation = 'PERTAHANKAN STOK'
        action_detail = 'Pertahankan level stok saat ini'
    else:
        category = 'Perlu Evaluasi'
        emoji = '❌'
        css_class = 'rec-tidak'
        recommendation = 'KURANGI STOK'
        action_detail = 'Kurangi stok 30-50%'
    
    return {
        'product_name': product_name,
        'score': score,
        'category': category,
        'emoji': emoji,
        'css_class': css_class,
        'recommendation': recommendation,
        'metrics': {
            'sentiment': avg_sentiment,
            'sales_proba': sales_proba,
            'positive_rate': positive_rate,
            'avg_rating': avg_rating,
            'review_count': review_count
        },
        'components': {
            'sentiment': sentiment_comp,
            'sales': sales_comp,
            'positive': positive_comp,
            'review': review_comp
        },
        'reviews': product_reviews
    }

# ==========================================
# MAIN APP
# ==========================================

def main():
    # Modern Header
    st.markdown("""
    <div class="gradient-header">
        <h1>📦 Product Sales Recommendation System</h1>
        <p>AI-Powered Product Analysis with Deep Neural Network & XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner('⏳ Loading AI models...'):
            sentiment_model, vectorizer, sales_model, error = load_models()
            
            if error:
                st.error(f"❌ Error loading models: {error}")
                st.info("""
                **Please ensure these files exist:**
                - `best_sentiment_model.h5`
                - `tfidf_vectorizer.pkl`
                - `xgboost_sales_classifier.pkl`
                """)
                return
            
            st.session_state.sentiment_model = sentiment_model
            st.session_state.vectorizer = vectorizer
            st.session_state.sales_model = sales_model
            st.session_state.models_loaded = True
    
    # ==========================================
    # SIDEBAR - HOW TO USE
    # ==========================================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #667eea; font-size: 2rem; margin: 0;">📦</h1>
            <h3 style="color: #667eea; margin: 0.5rem 0;">Product Recommendation</h3>
            <p style="color: #888; font-size: 0.85rem; margin: 0;">AI-Powered Analysis System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # How to Use Section
        st.markdown("### 📖 How to Use")
        
        with st.expander("🚀 **Quick Start**", expanded=True):
            st.markdown("""
            **Get started in 3 steps:**
            
            1. 📤 **Upload Dataset**
               - Go to "Product Analysis" tab
               - Click "Browse files" or drag & drop
               - Upload your CSV file
            
            2. 🔍 **Select Product**
               - Choose product from dropdown
               - Click "ANALYZE PRODUCT"
            
            3. 📊 **View Results**
               - See recommendation score
               - Check detailed metrics
               - Download report
            """)
        
        with st.expander("📋 **Dataset Requirements**"):
            st.markdown("""
            **Required Columns:**
            
            - `Nama Produk` - Product name
            - `comment-` - Customer reviews (Indonesian)
            - `rating` - Rating (1-5)
            
            **Format:**
            - File type: CSV
            - Max size: 200MB
            - Encoding: UTF-8
            
            **Example:**
            ```
            Nama Produk, comment-, rating
            iPhone 13, Bagus banget!, 5
            Samsung S21, Cukup oke, 4
            ```
            """)
        
        with st.expander("🎯 **Understanding Results**"):
            st.markdown("""
            **Recommendation Categories:**
            
            ✅ **Produk Unggulan** (≥75%)
            - Performa sangat baik
            - Action: Perbanyak stok 30-50%
            
            ⚠️ **Produk Stabil** (50-74%)
            - Performa cukup baik
            - Action: Pertahankan stok
            
            ❌ **Perlu Evaluasi** (<50%)
            - Performa kurang optimal
            - Action: Kurangi stok 30-50%
            
            ---
            
            **Score Components:**
            - 40% - Sentiment Analysis (DNN)
            - 30% - Sales Potential (XGBoost)
            - 20% - Positive Review Rate
            - 10% - Review Count
            """)
        
        with st.expander("💡 **Tips & Best Practices**"):
            st.markdown("""
            **For Best Results:**
            
            ✓ Use products with 10+ reviews
            
            ✓ Ensure reviews are in Indonesian
            
            ✓ Remove spam/fake reviews
            
            ✓ Update data regularly
            
            **When Analyzing:**
            
            ✓ Check Statistics Dashboard for overview
            
            ✓ Compare similar products
            
            ✓ Consider seasonal trends
            
            ✓ Download reports for records
            """)
        
        with st.expander("❓ **FAQ**"):
            st.markdown("""
            **Q: How long does analysis take?**
            A: 2-5 seconds per product
            
            **Q: Can I analyze multiple products?**
            A: Yes! Use Statistics Dashboard
            
            **Q: What if score is borderline?**
            A: Check detailed metrics & reviews
            
            **Q: Can I export results?**
            A: Yes! Use download button
            
            **Q: Is Indonesian language required?**
            A: Yes, for accurate sentiment analysis
            """)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #888; font-size: 0.75rem;">
            <p style="margin: 0;">Product Recommendation System</p>
            <p style="margin: 0.5rem 0 0 0;">© 2025 - SN</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab Navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Product Analysis", "📊 Statistics Dashboard", "ℹ️ About System"])
    
    # ==========================================
    # TAB 1: PRODUCT ANALYSIS
    # ==========================================
    with tab1:
        st.markdown("## 📂 Upload Your Dataset")
        
        st.markdown("""
        <div class="info-box">
            <strong>Required columns:</strong> Nama Produk, comment-, rating<br>
            <strong>File format:</strong> CSV (max 200MB)
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing product reviews",
            type=['csv'],
            help="Upload your e-commerce product data",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                # Validate
                required_cols = [PRODUCT_COLUMN, REVIEW_COLUMN, RATING_COLUMN]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ Missing required columns: {', '.join(required_cols)}")
                    return
                
                # Preprocess
                with st.spinner('🔄 Processing data...'):
                    df['cleaned_review'] = df[REVIEW_COLUMN].apply(
                        lambda x: preprocess_text(x, use_stemming=False, use_stopword=True)
                    )
                    st.session_state.df = df
                    st.session_state.products = sorted(df[PRODUCT_COLUMN].unique())
                
                st.success(f"✅ Data loaded successfully! Found **{len(st.session_state.products)} products**")
                
                st.markdown("---")
                
                # Product Selection
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_product = st.selectbox(
                        "🎯 Select Product to Analyze:",
                        st.session_state.products,
                        index=0
                    )
                
                with col2:
                    total_reviews = len(df[df[PRODUCT_COLUMN] == selected_product])
                    st.metric("Total Reviews", total_reviews)
                
                # Analyze Button
                if st.button("🚀 ANALYZE PRODUCT", use_container_width=True):
                    
                    with st.spinner('🤖 AI is analyzing...'):
                        result = predict_recommendation(
                            selected_product,
                            st.session_state.df,
                            st.session_state.sentiment_model,
                            st.session_state.vectorizer,
                            st.session_state.sales_model
                        )
                        
                        if 'error' in result:
                            st.error(result['error'])
                            return
                    
                    st.markdown("---")
                    
                    # Main Recommendation Card
                    st.markdown(f"""
                    <div class="rec-card {result['css_class']}">
                        <h1 style="margin:0; font-size: 3rem;">{result['emoji']}</h1>
                        <h2 style="margin: 1rem 0;">{result['recommendation']}</h2>
                        <h3 style="margin: 0.5rem 0; opacity: 0.9;">{result['product_name']}</h3>
                        <p style="font-size: 2.5rem; font-weight: 700; margin: 1rem 0;">
                            {result['score']*100:.1f}%
                        </p>
                        <p style="opacity: 0.9;">Recommendation Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics Row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>⭐ Avg Rating</h3>
                            <p>{result['metrics']['avg_rating']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>💚 Sentiment</h3>
                            <p>{result['metrics']['sentiment']*100:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>👍 Positive Rate</h3>
                            <p>{result['metrics']['positive_rate']*100:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>📈 Sales Potential</h3>
                            <p>{result['metrics']['sales_proba']*100:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Score breakdown
                        comp_data = pd.DataFrame({
                            'Component': ['Sentiment\n(40%)', 'Sales\n(30%)', 
                                        'Positive\n(20%)', 'Reviews\n(10%)'],
                            'Value': [
                                result['components']['sentiment'] * 100,
                                result['components']['sales'] * 100,
                                result['components']['positive'] * 100,
                                result['components']['review'] * 100
                            ]
                        })
                        
                        fig = px.bar(
                            comp_data,
                            x='Component',
                            y='Value',
                            text='Value',
                            title='Score Component Breakdown',
                            color='Value',
                            color_continuous_scale='Purples'
                        )
                        
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(height=400, showlegend=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Rating distribution
                        rating_counts = result['reviews'][RATING_COLUMN].value_counts().sort_index()
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=rating_counts.index,
                                y=rating_counts.values,
                                marker_color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'],
                                text=rating_counts.values,
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title='Customer Rating Distribution',
                            xaxis_title='Rating',
                            yaxis_title='Count',
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample Reviews
                    with st.expander("💬 View Sample Reviews"):
                        sample = result['reviews'].head(5)[[REVIEW_COLUMN, RATING_COLUMN, 'sentiment_score']].copy()
                        sample.columns = ['Review', 'Rating', 'Sentiment Score']
                        sample['Sentiment Score'] = sample['Sentiment Score'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(sample, use_container_width=True)
                    
                    # Download Report
                    report_data = {
                        'Product': [result['product_name']],
                        'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                        'Score': [f"{result['score']*100:.2f}%"],
                        'Category': [result['category']],
                        'Recommendation': [result['recommendation']],
                        'Sentiment': [f"{result['metrics']['sentiment']*100:.1f}%"],
                        'Rating': [f"{result['metrics']['avg_rating']:.2f}"],
                        'Positive Rate': [f"{result['metrics']['positive_rate']*100:.0f}%"],
                        'Reviews': [result['metrics']['review_count']]
                    }
                    
                    csv = pd.DataFrame(report_data).to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=csv,
                        file_name=f"analysis_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        else:
            st.markdown("""
            <div class="upload-box">
                <h3>📤 Drag and drop your CSV file here</h3>
                <p>or click Browse files button above</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Please upload your dataset in the Product Analysis tab. Statistics will be automatically generated.**
            
            **What you'll get:**
            - ✅ Instant product recommendation (Layak/Cukup/Tidak Layak Dijual)
            - 📊 Detailed sentiment & sales analysis
            - 📈 Interactive visualizations
            - 📥 Downloadable reports
            """)
    
    # ==========================================
    # TAB 2: STATISTICS DASHBOARD
    # ==========================================
    with tab2:
        st.markdown("## 📊 Product Statistics Overview")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Auto-analyze all products
            with st.spinner('🔄 Analyzing all products...'):
                all_results = []
                
                for product in st.session_state.products:
                    try:
                        result = predict_recommendation(
                            product,
                            df,
                            st.session_state.sentiment_model,
                            st.session_state.vectorizer,
                            st.session_state.sales_model
                        )
                        if 'error' not in result:
                            all_results.append({
                                'product': product,
                                'score': result['score'],
                                'category': result['category']
                            })
                    except:
                        continue
                
                results_df = pd.DataFrame(all_results)
            
            # Count by category
            unggulan = len(results_df[results_df['category'] == 'Produk Unggulan'])
            stabil = len(results_df[results_df['category'] == 'Produk Stabil'])
            evaluasi = len(results_df[results_df['category'] == 'Perlu Evaluasi'])
            total = len(results_df)
            
            # Info box
            st.markdown(f"""
            <div class="info-box">
                Statistics available for <strong>{total} products</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Big Category Cards (like screenshot)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 3rem 2rem; border-radius: 1rem; text-align: center; 
                            color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h1 style="font-size: 4rem; margin: 0; font-weight: 700;">{unggulan}</h1>
                    <p style="font-size: 1.3rem; margin: 1rem 0 0 0; opacity: 0.95;">
                        ✅ Produk Unggulan
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 3rem 2rem; border-radius: 1rem; text-align: center; 
                            color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h1 style="font-size: 4rem; margin: 0; font-weight: 700;">{stabil}</h1>
                    <p style="font-size: 1.3rem; margin: 1rem 0 0 0; opacity: 0.95;">
                        ⚠️ Produk Stabil
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 3rem 2rem; border-radius: 1rem; text-align: center; 
                            color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h1 style="font-size: 4rem; margin: 0; font-weight: 700;">{evaluasi}</h1>
                    <p style="font-size: 1.3rem; margin: 1rem 0 0 0; opacity: 0.95;">
                        ❌ Perlu Evaluasi
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Category Distribution")
                
                # Category pie chart
                category_counts = results_df['category'].value_counts()
                
                colors = {
                    'Produk Unggulan': '#5cb85c',
                    'Produk Stabil': '#f0ad4e', 
                    'Perlu Evaluasi': '#d9534f'
                }
                
                fig = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=0.4,
                    marker=dict(colors=[colors.get(cat, '#667eea') for cat in category_counts.index]),
                    textinfo='label+percent',
                    textposition='auto'
                )])
                
                fig.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Score Distribution")
                
                # Score histogram
                fig = px.histogram(
                    results_df,
                    x='score',
                    nbins=20,
                    title='',
                    color='category',
                    color_discrete_map={
                        'Produk Unggulan': '#5cb85c',
                        'Produk Stabil': '#f0ad4e',
                        'Perlu Evaluasi': '#d9534f'
                    },
                    labels={'score': 'Recommendation Score', 'count': 'Number of Products'}
                )
                
                fig.update_layout(
                    showlegend=True,
                    height=400,
                    xaxis_title='Recommendation Score',
                    yaxis_title='Number of Products',
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Product List with Categories
            st.markdown("### 📋 Product List by Category")
            
            # Filter
            filter_cat = st.multiselect(
                "Filter by category:",
                ['Produk Unggulan', 'Produk Stabil', 'Perlu Evaluasi'],
                default=['Produk Unggulan', 'Produk Stabil', 'Perlu Evaluasi']
            )
            
            filtered_df = results_df[results_df['category'].isin(filter_cat)].copy()
            filtered_df['score'] = filtered_df['score'].apply(lambda x: f"{x*100:.1f}%")
            filtered_df = filtered_df.sort_values('score', ascending=False)
            filtered_df.columns = ['Product Name', 'Score', 'Category']
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Product Statistics",
                data=csv,
                file_name=f"product_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("📤 Please upload your dataset in the **Product Analysis** tab. Statistics will be automatically generated.")
    
    # ==========================================
    # TAB 3: ABOUT SYSTEM
    # ==========================================
    with tab3:
        st.markdown("## ℹ️ Tentang Sistem")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Apa Itu Sistem Ini?
            
            Sistem rekomendasi produk berbasis AI yang membantu Anda menganalisis kelayakan produk 
            berdasarkan review pelanggan dan data penjualan.
            
            **Kegunaan:**
            - ✅ Tentukan produk mana yang perlu diperbanyak stoknya
            - ✅ Identifikasi produk yang perlu dievaluasi
            - ✅ Hemat waktu dengan analisis otomatis
            - ✅ Keputusan berbasis data, bukan asumsi
            
            ---
            
            ### 🧠 Cara Kerja Sistem
            
            **Step 1: Analisis Sentimen**
            - AI membaca semua review produk
            - Mengidentifikasi sentimen positif/negatif
            - Akurasi: 97%
            
            **Step 2: Prediksi Penjualan**
            - Machine learning memprediksi potensi penjualan
            - Berdasarkan pola rating dan review
            - Akurasi: 75%
            
            **Step 3: Skor Gabungan**
            - Kombinasi sentimen + prediksi penjualan + data rating
            - Menghasilkan skor rekomendasi 0-100%
            
            ---
            
            ### 📊 Kategori Produk
            
            """)
            
            # Category cards
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 0.8rem; color: white; margin: 1rem 0;">
                <h4 style="margin: 0;">✅ Produk Unggulan (Skor ≥ 75%)</h4>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Produk berkinerja sangat baik dengan review positif dan potensi penjualan tinggi.
                    <br><strong>Rekomendasi:</strong> Perbanyak stok 30-50%
                </p>
            </div>
            
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 0.8rem; color: white; margin: 1rem 0;">
                <h4 style="margin: 0;">⚠️ Produk Stabil (Skor 50-74%)</h4>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Produk dengan performa cukup baik, stabil tapi tidak outstanding.
                    <br><strong>Rekomendasi:</strong> Pertahankan stok saat ini
                </p>
            </div>
            
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 0.8rem; color: white; margin: 1rem 0;">
                <h4 style="margin: 0;">❌ Perlu Evaluasi (Skor < 50%)</h4>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Produk dengan performa rendah, review negatif, atau potensi penjualan kecil.
                    <br><strong>Rekomendasi:</strong> Kurangi stok 30-50% atau hentikan
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("""
            ### 📐 Formula Skor Rekomendasi
            
            Skor dihitung dari 4 komponen:
            
            | Komponen | Bobot | Penjelasan |
            |----------|-------|------------|
            | 💚 **Sentimen AI** | 40% | Analisis review menggunakan Deep Neural Network |
            | 📈 **Potensi Penjualan** | 30% | Prediksi kategori penjualan dengan XGBoost |
            | 👍 **Rating Positif** | 20% | Persentase review dengan rating 4-5 bintang |
            | 💬 **Jumlah Review** | 10% | Indikator reliabilitas data |
            
            **Total Skor = (Sentimen × 0.4) + (Penjualan × 0.3) + (Rating × 0.2) + (Review × 0.1)**
            
            ---
            
            ### ❓ Pertanyaan Umum (FAQ)
            
            **Q: Berapa lama waktu analisis?**  
            A: 2-5 detik per produk
            
            **Q: Apakah bisa menganalisis banyak produk sekaligus?**  
            A: Ya! Gunakan tab "Statistics Dashboard"
            
            **Q: Minimal berapa review yang dibutuhkan?**  
            A: Minimal 3-5 review, tapi semakin banyak semakin akurat
            
            **Q: Apakah harus bahasa Indonesia?**  
            A: Ya, sistem dioptimalkan untuk review berbahasa Indonesia
            
            **Q: Bagaimana jika skor borderline (mendekati 75% atau 50%)?**  
            A: Lihat detail metrik dan baca sample review untuk keputusan lebih baik
            
            **Q: Apakah hasil bisa didownload?**  
            A: Ya, setiap analisis bisa didownload dalam format CSV
            
            **Q: Data saya aman?**  
            A: Ya, data hanya diproses sementara dan tidak disimpan di server
            """)
        
        with col2:
            st.markdown("### 🎓 Teknologi")
            
            st.markdown("""
            <div class="metric-box">
                <h3>🤖 AI Models</h3>
                <p style="font-size: 1rem; margin-top: 0.5rem;">
                    Deep Neural Network<br>
                    <span style="color: #888; font-size: 0.9rem;">Sentiment Analysis</span>
                </p>
                <p style="font-size: 1rem; margin-top: 0.5rem;">
                    XGBoost ML<br>
                    <span style="color: #888; font-size: 0.9rem;">Sales Prediction</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 📊 Performance")
            
            st.markdown("""
            <div class="metric-box">
                <h3>Akurasi Sentimen</h3>
                <p>97.23%</p>
            </div>
            
            <div class="metric-box">
                <h3>Akurasi Penjualan</h3>
                <p>75.74%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 💡 Tips Penggunaan")
            
            st.info("""
            **Untuk Hasil Terbaik:**
            
            ✓ Gunakan produk dengan minimal 10+ review
            
            ✓ Pastikan review dalam bahasa Indonesia
            
            ✓ Hapus review spam/fake sebelum upload
            
            ✓ Update data secara berkala
            
            ✓ Bandingkan dengan produk sejenis
            """)
            
            st.markdown("---")
            
            st.markdown("### 📧 Bantuan")
            
            st.markdown("""
            Butuh bantuan atau ada pertanyaan?
            
            Gunakan panduan **"How to Use"** di sidebar →
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="custom-footer">
        <p><strong>📦 Product Sales Recommendation System</strong></p>
        <p>AI-Powered Product Analysis for E-Commerce</p>
        <p style="font-size: 0.8rem; color: #999; margin-top: 1rem;">
            © 2025 - Powered by Deep Neural Network & XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()