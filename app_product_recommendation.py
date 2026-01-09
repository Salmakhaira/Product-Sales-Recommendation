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
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# ==========================================
# TEXT PREPROCESSING
# ==========================================
def preprocess_text(text, use_stemming=False, use_stopword=True):
    """Preprocess text for sentiment analysis"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|<.*?>|\S+@\S+|@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) == 0:
        return ''
    
    if use_stopword:
        try:
            factory = StopWordRemoverFactory()
            remover = factory.create_stop_word_remover()
            text = remover.remove(text)
        except:
            pass
    
    if use_stemming:
        try:
            stemmer = StemmerFactory().create_stemmer()
            text = stemmer.stem(text)
        except:
            pass
    
    return text.strip()

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    """Load all trained models with flexible filename support"""
    sentiment_model = None
    vectorizer = None
    sales_model = None
    errors = []
    
    # Try loading sentiment model with various possible filenames
    sentiment_files = ['sentiment_dnn_model.h5', 'best_sentiment_model.h5', 'sentiment_model.h5']
    for filename in sentiment_files:
        try:
            sentiment_model = load_model(filename)
            break
        except Exception as e:
            errors.append(f"Sentiment model ({filename}): {str(e)}")
    
    # Try loading vectorizer
    vectorizer_files = ['tfidf_vectorizer.pkl', 'vectorizer.pkl']
    for filename in vectorizer_files:
        try:
            vectorizer = joblib.load(filename)
            break
        except Exception as e:
            errors.append(f"Vectorizer ({filename}): {str(e)}")
    
    # Try loading sales model
    sales_files = ['best_sales_classifier.pkl']
    for filename in sales_files:
        try:
            sales_model = joblib.load(filename)
            break
        except Exception as e:
            errors.append(f"Sales model ({filename}): {str(e)}")
    
    # Check if all models loaded successfully
    if sentiment_model is None or vectorizer is None or sales_model is None:
        error_msg = "Failed to load one or more models:\n" + "\n".join(errors)
        return None, None, None, error_msg
    
    return sentiment_model, vectorizer, sales_model, None

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def adjust_sentiment(rating, sentiment):
    """Adjust sentiment score based on rating"""
    if rating <= 1:
        return min(sentiment, 0.35)
    elif rating <= 2:
        return min(sentiment, 0.50)
    elif rating == 3:
        return max(min(sentiment, 0.70), 0.40)
    elif rating == 4:
        return max(min(sentiment, 0.90), 0.65)
    else:
        return min(max(sentiment, 0.80), 0.95)

def predict_recommendation(product_name, df, sentiment_model, vectorizer, sales_model, 
                          product_col='Nama Produk', review_col='comment-', rating_col='rating'):
    """Predict product recommendation using hybrid approach"""
    
    reviews = df[df[product_col] == product_name].copy()
    
    if len(reviews) == 0:
        return {'error': f"Product '{product_name}' not found"}
    
    # Remove rows with missing ratings
    reviews = reviews.dropna(subset=[rating_col])
    
    if len(reviews) == 0:
        return {'error': f"Product '{product_name}' has no valid ratings"}
    
    # Step 1: Sentiment prediction with DNN
    reviews_tfidf = vectorizer.transform(reviews['cleaned_review'])
    raw_scores = sentiment_model.predict(reviews_tfidf.toarray(), verbose=0).flatten()
    
    # Adjust sentiment based on rating
    reviews['sentiment_score'] = [
        adjust_sentiment(r, s) 
        for r, s in zip(reviews[rating_col], raw_scores)
    ]
    
    # Calculate features with robust NaN handling
    avg_sentiment = float(reviews['sentiment_score'].mean())
    std_sentiment = float(reviews['sentiment_score'].std())
    
    # Handle NaN in sentiment features
    if pd.isna(avg_sentiment):
        avg_sentiment = 0.5  # Default neutral sentiment
    if pd.isna(std_sentiment) or std_sentiment == 0:
        std_sentiment = 0.1
    
    # Calculate rating features
    positive_rate = float((reviews[rating_col] >= 4).mean())
    avg_rating = float(reviews[rating_col].mean())
    std_rating = float(reviews[rating_col].std())
    
    # Handle NaN in rating features
    if pd.isna(positive_rate):
        positive_rate = 0.5
    if pd.isna(avg_rating):
        avg_rating = 3.0  # Default average rating
    if pd.isna(std_rating) or std_rating == 0:
        std_rating = 0.1
    
    review_count = len(reviews)
    
    # Step 2: Sales prediction with XGBoost
    feature_df = pd.DataFrame([{
        'avg_sentiment': avg_sentiment,
        'std_sentiment': std_sentiment,
        'positive_rate': positive_rate,
        'avg_rating': avg_rating,
        'std_rating': std_rating,
        'review_count': review_count
    }])
    
    # Ensure no NaN values before prediction
    if feature_df.isna().any().any():
        feature_df = feature_df.fillna(0.5)
    
    sales_pred = int(sales_model.predict(feature_df)[0])
    sales_proba = float(sales_model.predict_proba(feature_df)[0, 1])
    
    # Step 3: Calculate hybrid score
    sentiment_comp = avg_sentiment * 0.4
    sales_comp = sales_proba * 0.2
    positive_comp = positive_rate * 0.3
    review_comp = (min(review_count, 100) / 100.0) * 0.1
    
    score = min(sentiment_comp + sales_comp + positive_comp + review_comp, 1.0)
    
    # Categorize
    if score >= 0.75:
        category = 'Produk Unggulan'
    elif score >= 0.5:
        category = 'Produk Stabil'
    else:
        category = 'Perlu Evaluasi'
    
    return {
        'product_name': product_name,
        'recommendation_score': score,
        'recommendation_category': category,
        'sentiment_avg': avg_sentiment,
        'positive_rate': positive_rate,
        'review_count': review_count,
        'sales_category': 'High Seller' if sales_pred == 1 else 'Low Seller',
        'sales_proba': sales_proba
    }

# ==========================================
# MAIN APP
# ==========================================
def main():
    # Auto-load models on first run
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            sentiment_model, vectorizer, sales_model, error = load_models()
            if error is None:
                st.session_state.sentiment_model = sentiment_model
                st.session_state.vectorizer = vectorizer
                st.session_state.sales_model = sales_model
                st.session_state.models_loaded = True
                st.session_state.model_error = None
            else:
                st.session_state.model_error = error
    
    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>📦 Product Recommendation System</h1>
        <p>AI-Powered Product Analysis for E-Commerce</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        if st.session_state.models_loaded:
            st.success("🟢 Ready")
        else:
            st.error("🔴 Models Not Loaded")
            st.warning("Please check the main tab for details")
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload your CSV dataset
        2. Analyze products
        """)

    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Product Analysis", "📈 Statistics Dashboard", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Product Analysis")
        
        if not st.session_state.models_loaded:
            st.error("❌ Failed to load required model files")
            
            if hasattr(st.session_state, 'model_error'):
                st.error(st.session_state.model_error)
            
            st.info("""
            **Please ensure you have the following model files in the same directory as this app:**
            
            **Option 1 (Primary names):**
            - `sentiment_dnn_model.h5` (Sentiment Analysis Model)
            - `tfidf_vectorizer.pkl` (Text Vectorizer)
            - `xgboost_sales_model.pkl` (Sales Prediction Model)
            
            **Option 2 (Alternative names):**
            - `best_sentiment_model.h5`
            - `vectorizer.pkl`
            - `best_sales_classifier.pkl`
            
            The system will automatically try to load any of these filenames.
            """)
            return
        
        uploaded_file = st.file_uploader(
            "Upload your product dataset (CSV)",
            type=['csv'],
            help="Dataset should contain: Product Name, Reviews, and Ratings"
        )
        
        if uploaded_file:
            # Check if this is a new file - if yes, reset statistics
            current_file_name = uploaded_file.name
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != current_file_name:
                st.session_state.results_df = None  # Reset statistics
                st.session_state.last_uploaded_file = current_file_name
                
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success(f"✅ Dataset loaded: {len(df)} reviews")
            
            st.markdown("---")
            
            # Fixed column names
            product_col = 'Nama Produk'
            review_col = 'comment-'
            rating_col = 'rating'
            
            # Single product analysis
            st.markdown("### 🔍 Single Product Analysis")
            
            unique_products = df[product_col].unique()
            selected_product = st.selectbox(
                "Select a product to analyze:",
                unique_products,
                help="Choose a product from your dataset"
            )
            
            if st.button("🚀 Analyze Product", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Preprocess
                    df['cleaned_review'] = df[review_col].apply(
                        lambda x: preprocess_text(x, use_stemming=False, use_stopword=True)
                    )
                    
                    # Predict
                    result = predict_recommendation(
                        selected_product, df,
                        st.session_state.sentiment_model,
                        st.session_state.vectorizer,
                        st.session_state.sales_model,
                        product_col, review_col, rating_col
                    )
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        # Display results
                        st.markdown("### 📊 Analysis Results")
                        
                        score = result['recommendation_score']
                        category = result['recommendation_category']
                        
                        if category == 'Produk Unggulan':
                            card_class = 'rec-layak'
                            emoji = '✅'
                            color = '#5cb85c'
                        elif category == 'Produk Stabil':
                            card_class = 'rec-cukup'
                            emoji = '⚠️'
                            color = '#f0ad4e'
                        else:
                            card_class = 'rec-tidak'
                            emoji = '❌'
                            color = '#d9534f'
                        
                        st.markdown(f"""
                        <div class="rec-card {card_class}">
                            <h2 style="margin: 0;">{emoji} {category}</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;">{score*100:.1f}%</h1>
                            <p style="font-size: 1.2rem; opacity: 0.9;">Recommendation Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Sentiment Score", f"{result['sentiment_avg']*100:.1f}%")
                        with col2:
                            st.metric("Positive Reviews", f"{result['positive_rate']*100:.1f}%")
                        with col3:
                            st.metric("Total Reviews", result['review_count'])
                        with col4:
                            st.metric("Sales Category", result['sales_category'])
                        
                        # Sample reviews
                        st.markdown("---")
                        
                                                # Simple recommendations
                        if result['recommendation_category'] == 'Produk Unggulan':
                            st.success("""
                            **💡 Rekomendasi:**
                            - Produk ini memiliki tingkat penerimaan yang sangat tinggi dari pelanggan dan dapat dijadikan benchmark dalam pengembangan produk baru dengan karakteristik serupa.
                            - Fitur serta karakteristik produk terbukti unggul. Perlu dilakukan analisis mendalam terhadap faktor-faktor utama yang mendorong keberhasilan produk ini.
                            - Kualitas produk perlu dipertahankan, serta formula keberhasilannya dapat direplikasi dan diadaptasi pada pengembangan produk lain.
                            """)
                        elif result['recommendation_category'] == 'Produk Stabil':
                            st.info("""
                            **💡 Rekomendasi:**
                            - Produk ini menunjukkan performa yang cukup baik, namun masih memiliki ruang untuk pengembangan dan penyempurnaan.
                            - Lakukan perbandingan dengan produk unggulan (top-performing products) untuk mengidentifikasi aspek yang masih dapat ditingkatkan.
                            - roduk ini dapat dimanfaatkan sebagai objek eksperimen untuk pengujian peningkatan fitur maupun kualitas produk.
                            """)
                        else:  # Perlu Evaluasi
                            st.warning("""
                            **💡 Rekomendasi:**
                            - Lakukan evaluasi menyeluruh terhadap ulasan negatif pelanggan untuk mengidentifikasi penyebab utama rendahnya minat pasar.
                            - Hindari pengembangan produk baru dengan karakteristik yang serupa sebelum dilakukan perbaikan signifikan.
                            - Evaluasi kesesuaian produk terhadap target pasar, serta pertimbangkan kebutuhan perubahan strategis yang bersifat fundamental.
                            """)
                        
                        st.markdown("### 💬 Sample Reviews")
                        
                        product_reviews = df[df[product_col] == selected_product][[review_col, rating_col]].head(5)
                        
                        for idx, row in product_reviews.iterrows():
                            rating = int(row[rating_col])
                            stars = "⭐" * rating
                            st.markdown(f"**{stars}** {rating}/5")
                            st.write(row[review_col])
                            st.markdown("---")
    
    with tab2:
        st.markdown("## Statistics Dashboard")
        
        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            # Summary cards
            col1, col2, col3 = st.columns(3)
            
            unggulan = len(results_df[results_df['category'] == 'Produk Unggulan'])
            stabil = len(results_df[results_df['category'] == 'Produk Stabil'])
            evaluasi = len(results_df[results_df['category'] == 'Perlu Evaluasi'])
            
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
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
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
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
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
            
            # Top 10 Products
            st.markdown("### 🏆 Top 10 Best Products")
            
            # Get top 10 products by score
            top_10_df = results_df.nlargest(10, 'score').copy()
            top_10_df['score_display'] = top_10_df['score'].apply(lambda x: f"{x*100:.1f}%")
            
            # Prepare display dataframe
            display_df = top_10_df[['product_name', 'score_display', 'category']].reset_index(drop=True)
            display_df.index = display_df.index + 1  # Start index from 1
            display_df.columns = ['Product Name', 'Score', 'Category']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download full results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Statistics",
                data=csv,
                file_name=f"statistics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("📊 Analyze all products to view statistics")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                product_col = 'Nama Produk'
                review_col = 'comment-'
                rating_col = 'rating'
                
                unique_products = df[product_col].unique()
                
                st.write(f"**Dataset loaded:** {len(df)} reviews from {len(unique_products)} products")
                
                if st.button("🚀 Analyze All Products", use_container_width=True, key="batch_analyze"):
                    with st.spinner("Processing all products..."):
                        df['cleaned_review'] = df[review_col].apply(
                            lambda x: preprocess_text(x, use_stemming=False, use_stopword=True)
                        )
                        
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, product in enumerate(unique_products):
                            result = predict_recommendation(
                                product, df,
                                st.session_state.sentiment_model,
                                st.session_state.vectorizer,
                                st.session_state.sales_model,
                                product_col, review_col, rating_col
                            )
                            
                            if 'error' not in result:
                                results.append({
                                    'product_name': result['product_name'],
                                    'score': result['recommendation_score'],
                                    'category': result['recommendation_category']
                                })
                            
                            progress_bar.progress((idx + 1) / len(unique_products))
                        
                        results_df = pd.DataFrame(results)
                        st.session_state.results_df = results_df
                        
                        st.success(f"✅ Analysis complete! {len(results)} products analyzed")
                        st.rerun()
            else:
                st.warning("⚠️ Please upload dataset first in the Product Analysis tab")
    
    with tab3:
        st.markdown("## About System")
        
        st.markdown("""
        ### 🎯 System Purpose
        
        AI-powered product recommendation system that analyzes product performance based on customer reviews and sales data.
        
        ### 🧠 How It Works
        
        **Step 1: Sentiment Analysis**
        - Deep Neural Network analyzes customer reviews
        - Identifies positive/negative sentiments
        
        **Step 2: Sales Prediction**
        - XGBoost model predicts sales potential
        - Based on historical patterns and ratings
        
        **Step 3: Hybrid Score**
        - Combines sentiment + sales prediction + ratings
        - Generates recommendation score (0-100%)
        
        ### 📊 Product Categories
        
        **✅ Produk Unggulan (Score ≥ 75%)**
        - High performance with positive reviews
        - Recommendation: Increase stock by 30-50%
        
        **⚠️ Produk Stabil (Score 50-74%)**
        - Decent performance, stable but not outstanding
        - Recommendation: Maintain current stock
        
        **❌ Perlu Evaluasi (Score < 50%)**
        - Low performance with negative reviews
        - Recommendation: Reduce stock by 30-50% or discontinue
        
        ### 📐 Score Formula
        
        **Total Score = (Sentiment × 0.4) + (Positive Reviews × 0.3) + (Sales × 0.2) + (Review Count × 0.1)**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p><strong>📦 Product Recommendation System</strong> | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
