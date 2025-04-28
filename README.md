# App Review Sentiment Analysis Dashboard

This project provides a Streamlit dashboard for analyzing sentiment and exploring insights from app reviews. It visualizes key metrics, sentiment trends, and common topics mentioned in user feedback.

## Features

*   **Interactive Dashboard:** Built with Streamlit for easy filtering and exploration.
*   **Key Performance Indicators (KPIs):**
    *   Positive Sentiment %
    *   Average Star Rating
    *   Total Reviews
    *   Neutral Reviews
    *   Users Rated
    *   Unique Review Days
*   **Visualizations:**
    *   Review Count by Sentiment (Bar Chart)
    *   Star Rating Distribution (Bar Chart)
    *   Positive Sentiment % Over Time (Line Chart)
    *   Average Rating by App Version (Bar Chart - if available)
    *   Top Positive/Negative Keywords (Horizontal Bar Charts - expandable)
    *   Review Labels Distribution (Bar Chart - if available, expandable)
    *   Top Feature Request Keywords (Horizontal Bar Chart - if available, expandable)
*   **Data Filtering:** Filter reviews by date range and app version (if available).
*   **Data Download:** Download the filtered review data as a CSV file.

## Data

The dashboard uses app review data provided in a CSV file:

This file should contain columns like `at` (timestamp), `score` (rating), `review` (text), `sentiment` (optional, will be calculated if missing), `appVersion` (optional), and `label` (optional).

## Setup

1.  **Clone the repository (or ensure you have the project files).**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    pandas
    textblob
    scikit-learn
    plotly
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK corpora (needed for TextBlob):**
    Run the following command in your terminal after activating the virtual environment:
    ```bash
    python -m textblob.download_corpora
    ```

## Running the App

1.  Make sure your virtual environment is activated.
2.  Ensure the data file (`bq-results-20250422-112433-1745321081904.csv`) is in the same directory as `app_dashboard.py`.
3.  Run the Streamlit application:
    ```bash
    streamlit run app_dashboard.py
    ```
4.  The dashboard will open in your default web browser.

## Potential Improvements

*   More sophisticated NLP for keyword extraction (e.g., TF-IDF, topic modeling).
*   Error handling for different CSV column names or formats.
*   Additional visualizations (e.g., sentiment distribution per version).
*   More advanced text cleaning before sentiment analysis.
