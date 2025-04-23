import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go 


st.set_page_config(
    page_title="App Review Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Improve metric visuals */
    [data-testid="stMetricValue"] {
        font-size: 2em;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1em;
        color: gray;
    }
    /* Consistent plot height for Plotly */
    .stPlotlyChart {
        height: 400px;
    }
    h1, h2, h3 {
        color: #2c3e50; /* Darker headings */
    }
    .block-container { /* Add some padding */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 App Review Sentiment Analysis Dashboard")
st.markdown("Explore insights from app reviews, analyze sentiment trends, and identify key topics.")

@st.cache_data
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Loads, prepares, and calculates sentiment if needed."""
    try:
        df = pd.read_csv(file_path)
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df['review'] = df['review'].fillna("")
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['at', 'score'], inplace=True)
        df['score'] = df['score'].astype(int)

        if 'sentiment' not in df.columns or df['sentiment'].isnull().any():
            st.info("Calculating sentiment using TextBlob...", icon="⚙️")
            df['sentiment'] = df['review'].apply(classify_sentiment)
            st.success("Sentiment calculation complete.", icon="✅")

        return df
    except FileNotFoundError:
        st.error(f"Fatal Error: The data file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Fatal Error: An error occurred while loading the data: {e}")
        return pd.DataFrame()

def classify_sentiment(text: str) -> str:
    """Classifies text sentiment using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1: return 'Positive'
    if polarity < -0.1: return 'Negative'
    return 'Neutral'

CSV_PATH = 'dataset.csv'
with st.spinner("Loading and preparing review data..."):
    df_raw = load_and_prepare_data(CSV_PATH)

if not df_raw.empty:

    st.sidebar.header("⚙️ Filters")

    min_date_available = df_raw['at'].min().date()
    max_date_available = df_raw['at'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date_available, max_date_available),
        min_value=min_date_available,
        max_value=max_date_available,
        help="Filter reviews based on their submission date."
    )
    start_date = date_range[0]
    end_date = date_range[1] if len(date_range) > 1 else date_range[0]


    selected_version = 'All'
    if 'appVersion' in df_raw.columns and df_raw['appVersion'].dropna().nunique() > 1:
        valid_versions = df_raw['appVersion'].dropna().astype(str).unique()
        versions_list = ['All'] + sorted(list(valid_versions))
        selected_version = st.sidebar.selectbox(
            'Filter by App Version',
            versions_list,
            help="Select a specific app version or 'All'."
        )


    filtered_df = df_raw[
        (df_raw['at'].dt.date >= start_date) &
        (df_raw['at'].dt.date <= end_date)
    ]
    if selected_version != 'All' and 'appVersion' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['appVersion'].astype(str) == selected_version]


    if filtered_df.empty:
        st.warning("No data available for the selected filters. Try adjusting the date range or version.", icon="⚠️")
    else:
        st.subheader("📈 Key Performance Indicators")

        total_reviews = len(filtered_df)
        pos_count = (filtered_df['sentiment'] == 'Positive').sum()
        neg_count = (filtered_df['sentiment'] == 'Negative').sum()
        neut_count = total_reviews - pos_count - neg_count
        positive_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
        avg_rating = filtered_df['score'].mean() if total_reviews > 0 else 0
        users_reviewed = filtered_df['at'].dt.date.nunique()
        users_rated = filtered_df['score'].count()


        cols = st.columns(6)
        cols[0].metric("Positive Sentiment %", f"{positive_pct:.1f}%")
        cols[1].metric("Average Rating", f"{avg_rating:.2f}")
        cols[2].metric("Total Reviews", f"{total_reviews:,}")
        cols[3].metric("Neutral Reviews", f"{neut_count:,}")
        cols[4].metric("Users Rated", f"{users_rated:,}")
        cols[5].metric("Unique Review Days", f"{users_reviewed:,}")

        st.markdown("---")

        st.subheader("📊 Core Sentiment & Rating Metrics")
        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Review Count by Sentiment")
            sentiment_counts = filtered_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
            fig_sentiment = px.bar(
                sentiment_counts,
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
                color=sentiment_counts.index,
                color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'}, # Updated colors

            )
            fig_sentiment.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig_sentiment, use_container_width=True)

            st.markdown("#### Positive Sentiment % Over Time")
            time_series = filtered_df.set_index('at').resample('W')['sentiment'].apply(
                lambda x: (x == 'Positive').sum() / len(x) * 100 if len(x) > 0 else 0
            ).rename('Positive %')
            fig_time_series = px.line(
                time_series,
                y='Positive %',
                labels={'at': 'Week', 'Positive %': 'Positive Sentiment (%)'},
            )
            fig_time_series.update_layout(title_x=0.5)
            st.plotly_chart(fig_time_series, use_container_width=True)

        with col2:
            st.markdown("#### Star Rating Distribution")
            rating_dist = filtered_df['score'].value_counts().sort_index()
            fig_rating = px.bar(
                rating_dist,
                x=rating_dist.index,
                y=rating_dist.values,
                labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
                color=rating_dist.index, 
                color_continuous_scale=px.colors.sequential.YlOrRd, 
            )
            fig_rating.update_layout(coloraxis_showscale=False, title_x=0.5)
            st.plotly_chart(fig_rating, use_container_width=True)

            if 'appVersion' in filtered_df.columns and selected_version == 'All' and filtered_df['appVersion'].nunique() > 1:
                st.markdown("#### Average Rating by App Version")
                ver_avg = filtered_df.groupby(filtered_df['appVersion'].astype(str))['score'].mean().sort_values(ascending=False).round(2)
                fig_ver_avg = px.bar(
                    ver_avg,
                    x=ver_avg.index,
                    y=ver_avg.values,
                    labels={'x': 'App Version', 'y': 'Average Rating'},
                )
                fig_ver_avg.update_layout(title_x=0.5)
                st.plotly_chart(fig_ver_avg, use_container_width=True)
            elif 'appVersion' not in filtered_df.columns:
                 st.info("App Version data not available for comparison.", icon="ℹ️")

        st.markdown("---")
        with st.expander("🔑 Explore Top Keywords in Reviews", expanded=False):

            def get_top_keywords(texts, n=15, stop_words='english'):
                """Extracts top N keywords using CountVectorizer."""
                if texts.empty: return []
                try:
                    vec = CountVectorizer(stop_words=stop_words).fit(texts)
                    dtm = vec.transform(texts)
                    freqs = dtm.sum(axis=0).A1
                    words = vec.get_feature_names_out()
                    word_freq = list(zip(words, freqs))
                    word_freq.sort(key=lambda x: x[1], reverse=True)
                    return word_freq[:n]
                except ValueError as e:
                    st.warning(f"Could not extract keywords (possibly due to only stop words found): {e}", icon="⚠️")
                    return []

            def plot_keywords_plotly(keywords, title, color):
                """Plots keywords using a Plotly horizontal bar chart."""
                if not keywords:
                    st.write(f"No keywords to display for '{title}'.")
                    return

                words, counts = zip(*keywords)
                fig = go.Figure(go.Bar(
                    y=words[::-1], 
                    x=counts[::-1], 
                    orientation='h', 
                    marker_color=color
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Frequency",
                    yaxis_title="Keyword",
                    yaxis=dict(tickmode='array', tickvals=list(range(len(words))), ticktext=words[::-1]),
                    margin=dict(l=100, r=20, t=50, b=50), 
                    height=max(300, len(words) * 25) 
                )
                st.plotly_chart(fig, use_container_width=True)

            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                neg_texts = filtered_df[filtered_df['sentiment'] == 'Negative']['review']
                neg_kw = get_top_keywords(neg_texts)
                plot_keywords_plotly(neg_kw, "Top Negative Keywords", "#e74c3c") 

            with kw_col2:
                pos_texts = filtered_df[filtered_df['sentiment'] == 'Positive']['review']
                pos_kw = get_top_keywords(pos_texts)
                plot_keywords_plotly(pos_kw, "Top Positive Keywords", "#2ecc71") 

        if 'label' in filtered_df.columns:
            st.markdown("---")
            with st.expander("🏷️ Analyze Review Labels and Feature Requests", expanded=False):
                label_col1, label_col2 = st.columns(2)

                with label_col1:
                    st.markdown("#### Distribution of Review Labels")
                    label_counts = filtered_df['label'].dropna().astype(str).value_counts()
                    if not label_counts.empty:
                        fig_labels = px.bar(
                            label_counts,
                            x=label_counts.index,
                            y=label_counts.values,
                            labels={'x': 'Label', 'y': 'Count'},
                        )
                        fig_labels.update_layout(title_x=0.5)
                        st.plotly_chart(fig_labels, use_container_width=True)
                    else:
                        st.write("No labels found in the filtered data.")

                with label_col2:
                    st.markdown("#### Top Keywords in Feature Requests")
                    feat_texts = filtered_df[filtered_df['label'].astype(str).str.contains('feature', na=False, case=False)]['review']
                    if not feat_texts.empty:
                        feat_kw = get_top_keywords(feat_texts)
                        plot_keywords_plotly(feat_kw, "Top Feature Request Keywords", "#3498db") # Blue
                    else:
                        st.write("No reviews explicitly labeled as 'feature request' found.")
        else:
            st.info("Label column not found. Skipping Label Analysis.", icon="ℹ️")

        st.markdown("---")
        st.subheader("📥 Download Filtered Data")
        st.markdown("Get the currently displayed review data in CSV format.")
        try:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name=f"filtered_reviews_{start_date}_to_{end_date}{'_'+selected_version if selected_version != 'All' else ''}.csv",
                mime='text/csv',
                help="Download the review data based on the current filter settings."
            )
        except Exception as e:
            st.error(f"Could not prepare data for download: {e}")


else:
    st.error("Dashboard cannot be displayed. Please check the data file path and format.")

st.markdown("---")
st.caption("Dashboard created with Streamlit | Data Source: App Reviews") 