import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk import download

# --------------------------------------------------
# ⚙️ PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="European Football Sentiment Dashboard",
    page_icon="⚽",
    layout="wide"
)

# --------------------------------------------------
# 📂 LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    """Load and clean football sentiment dataset."""
    df = pd.read_csv("football_sentiments_all.csv")
    df["sentiment"] = df["sentiment"].str.capitalize()

    # Convert date and extract year
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Clean club names for better readability
    df["club_name"] = (
        df["club_name"]
        .str.replace(r"[_-]", " ", regex=True)
        .str.replace(r"FC|cf|official|en|es|cat|de|fr|nl", "", case=False, regex=True)
        .str.strip()
        .str.title()
    )
    return df

df = load_data()

# --------------------------------------------------
# 🎨 STYLE
# --------------------------------------------------
st.markdown("""
    <style>
        h1, h2, h3, h4 { color: #1E3D59; }
        .metric-label { font-weight:600; color:#555; }
        [data-testid="stMetricValue"] {
            color: #0077B6;
            font-weight:700;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 🧭 HEADER
# --------------------------------------------------
st.title("⚽ European Football Sentiment Dashboard")
st.markdown("""
Explore **fan sentiment** from 1,500 football tweets posted by official club accounts across Europe.  
Use filters to explore sentiment by **year** and **club**, and gain insights from trends, volumes, and fan language.
""")

# --------------------------------------------------
# 🎛️ SIDEBAR FILTERS
# --------------------------------------------------
years = sorted(df["year"].dropna().unique().astype(int))
selected_year = st.sidebar.selectbox("📆 Select Year", ["All"] + list(years))

clubs = sorted(df['club_name'].dropna().unique())
selected_club = st.sidebar.selectbox("🏟️ Select Club", ["All"] + clubs)

filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]
if selected_club != "All":
    filtered_df = filtered_df[filtered_df["club_name"] == selected_club]

st.sidebar.markdown("---")
st.sidebar.write(f"Displaying **{len(filtered_df)} tweets**")

# --------------------------------------------------
# 🧭 MULTI-DASHBOARD TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Sentiment Overview", "🏆 Club Comparison", "📈 Yearly Trends", "☁️ Word Cloud"]
)

# --------------------------------------------------
# TAB 1: SENTIMENT OVERVIEW
# --------------------------------------------------
with tab1:
    st.header("📊 Overall Sentiment Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", len(filtered_df))
    col2.metric("Positive", len(filtered_df[filtered_df['sentiment'] == 'Positive']))
    col3.metric("Negative", len(filtered_df[filtered_df['sentiment'] == 'Negative']))

    st.markdown("Explore the sentiment composition, club activity, and emotional balance across tweets.")

    # Sentiment Distribution Pie
    st.subheader("⚙️ Sentiment Composition")
    fig = px.pie(
        filtered_df,
        names="sentiment",
        color="sentiment",
        color_discrete_map={
            "Positive": "#00BFA6",
            "Negative": "#FF595E"
        },
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment Volume
    st.subheader("📈 Sentiment Volume Overview")
    sentiment_counts = (
        filtered_df.groupby("sentiment")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig_bar = px.bar(
        sentiment_counts,
        x="sentiment",
        y="count",
        text="count",
        color="sentiment",
        color_discrete_map={
            "Positive": "#00BFA6",
            "Negative": "#FF595E"
        },
        title="Volume of Tweets by Sentiment"
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Monthly Sentiment Trend
    st.subheader("🕓 Sentiment Over Time (Monthly Trends)")
    time_df = filtered_df.copy()
    time_df["month"] = time_df["date"].dt.to_period("M").astype(str)
    trend_df = time_df.groupby(["month", "sentiment"]).size().reset_index(name="count")
    fig_line = px.line(
        trend_df,
        x="month",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={
            "Positive": "#00BFA6",
            "Negative": "#FF595E"
        },
        title="Monthly Tweet Activity by Sentiment"
    )
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Number of Tweets")
    st.plotly_chart(fig_line, use_container_width=True)

    # Top 10 Clubs
    st.subheader("🏟️ Top 10 Clubs by Tweet Volume")
    club_rank = (
        filtered_df.groupby("club_name")
        .size()
        .reset_index(name="tweet_count")
        .sort_values("tweet_count", ascending=False)
        .head(10)
    )
    fig_club = px.bar(
        club_rank,
        x="tweet_count",
        y="club_name",
        orientation="h",
        color="tweet_count",
        color_continuous_scale="tealgrn",
        title="Most Active Clubs by Tweets",
        text="tweet_count"
    )
    fig_club.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig_club, use_container_width=True)

    # Sentiment Ratio Bubble
    st.subheader("⚽ Sentiment Balance Across Clubs")
    ratio_df = (
        filtered_df.groupby(["club_name", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if all(x in ratio_df.columns for x in ["Positive", "Negative"]):
        ratio_df["sentiment_ratio"] = ratio_df["Positive"] / (ratio_df["Negative"] + 1)
        ratio_df = ratio_df.sort_values("sentiment_ratio", ascending=False).head(20)

        fig_bubble = px.scatter(
            ratio_df,
            x="Positive",
            y="Negative",
            size="sentiment_ratio",
            hover_name="club_name",
            color="sentiment_ratio",
            color_continuous_scale="Bluered",
            title="Clubs: Positive vs Negative Sentiment Balance"
        )
        fig_bubble.update_layout(xaxis_title="Positive Tweets", yaxis_title="Negative Tweets")
        st.plotly_chart(fig_bubble, use_container_width=True)

# --------------------------------------------------
# TAB 2: CLUB COMPARISON
# --------------------------------------------------
with tab2:
    st.header("🏆 Sentiment Comparison Across Clubs")
    club_counts = filtered_df.groupby(['club_name', 'sentiment']).size().reset_index(name='count')

    fig2 = px.bar(
        club_counts,
        x="club_name",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={
            "Positive": "#00BFA6",
            "Negative": "#FF595E"
        },
        title="Club-wise Sentiment Breakdown"
    )
    fig2.update_layout(
        xaxis_title="Club",
        yaxis_title="Tweet Volume",
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#333")
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# TAB 3: YEARLY TRENDS
# --------------------------------------------------
with tab3:
    st.header("📈 Yearly Sentiment Trends")
    trend_df = df.groupby(['year', 'sentiment']).size().reset_index(name='count')

    fig3 = px.line(
        trend_df,
        x="year",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={
            "Positive": "#00BFA6",
            "Negative": "#FF595E"
        },
        title="Sentiment Trend Over the Years"
    )
    fig3.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Tweets",
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#333")
    )
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# TAB 4: CLEANED WORD CLOUD
# --------------------------------------------------
download('stopwords')
stop_words = set(stopwords.words('english'))
extra_stops = {
    'https', 'co', 'amp', 'fc', 'barca', 'juve', 'madrid', 'bayern', 'real', 'ucl',
    'team', 'game', 'match', 'season', 'goal', 'win', 'rt', 'vs', 'today', 'club',
    'de', 'la', 'el', 'en', 'con', 'un', 'los', 'las', 'para', 'del',
    'und', 'die', 'das', 'der', 'mit', 'auf', 'zum', 'que', 'je', 'au'
}
stop_words.update(extra_stops)

with tab4:
    st.header("☁️ Cleaned Word Cloud — Most Frequent Keywords in Tweets")
    text_data = " ".join(filtered_df['text'].astype(str))

    if len(text_data.strip()) == 0:
        st.warning("No text data for this selection.")
    else:
        # Clean and preprocess text
        text_data = text_data.lower()
        text_data = re.sub(r"http\S+", "", text_data)
        text_data = re.sub(r"@\w+", "", text_data)
        text_data = re.sub(r"[^a-zA-Z\s]", " ", text_data)
        text_data = " ".join(
            [w for w in text_data.split() if w not in stop_words and len(w) > 2]
        )

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="cool",
            collocations=False,
            max_words=100
        ).generate(text_data)

        st.image(wc.to_array(), use_column_width=True)
        st.markdown(
            """
            #### 💬 Interpretation
            - Common neutral and multilingual filler words were removed.  
            - The remaining terms highlight **emotional tone and topical focus** of fan discussions.  
            - Useful for exploring **what drives positive or negative engagement**.
            """
        )

# --------------------------------------------------
# SUMMARY INSIGHTS
# --------------------------------------------------
st.markdown("---")
st.subheader("💡 Insights & Observations")
st.markdown(
    """
    - **Overall sentiment** is slightly negative, often influenced by match outcomes.  
    - **Positive surges** appear during wins, transfers, or special announcements.  
    - **Smaller clubs** show proportionally higher fan positivity.  
    - Cleaned **word clouds** highlight emotional, event-driven language instead of noise.  
    """
)
