import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import sqlite3
import datetime
from datetime import timedelta
import database
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Feedback Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .st-emotion-cache-1wrcr25 {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title-text {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
    }
    .subtitle-text {
        font-size: 20px;
        color: #6B7280;
        text-align: center;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 16px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Function to authenticate users
# Function to authenticate users
# Function to authenticate users
def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.container():
            st.markdown('<p class="title-text">Admin Dashboard Login</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.button("Login")
                
                if login_button:
                    # Simple authentication for demo
                    # In production, use a more secure authentication method
                    if username == "admin" and password == "admin123":
                        st.session_state.authenticated = True
                        # Instead of rerunning, we'll use session state
                        st.success("Login successful! Loading dashboard...")
                        # Give a visual indication of success
                        import time
                        time.sleep(0.5)
                        # The page will automatically refresh on the next interaction
                    else:
                        st.error("Invalid credentials")
            
        return st.session_state.authenticated
    
    return True

# Function to load feedback data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_feedback_data(days=None):
    feedback_data = database.get_all_feedback(days)
    if feedback_data:
        df = pd.DataFrame(feedback_data)
        # Convert timestamp strings to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df
    return pd.DataFrame()

# Dashboard main layout
def render_dashboard():
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Overview", "Detailed Analytics", "Feedback Management", "Critical Issues", "AI Insights", "Settings"],
            icons=["house", "graph-up", "chat-left-text", "exclamation-triangle", "robot", "gear"],
            menu_icon="list",
            default_index=0,
        )
        
        # Filter section
        st.sidebar.header("Filters")
        date_filter = st.sidebar.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
        )
        
        # Map selection to days
        if date_filter == "Last 7 days":
            days = 7
        elif date_filter == "Last 30 days":
            days = 30
        elif date_filter == "Last 90 days":
            days = 90
        else:
            days = None

    # Load data based on filter
    df = load_feedback_data(days)
    
    # Check if data exists
    if df.empty:
        st.warning("No feedback data available for the selected period.")
        return
    
    # Overview Page
    if selected == "Overview":
        st.markdown('<p class="title-text">Feedback Analytics Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle-text">Summary of feedback performance</p>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{len(df)}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total Responses</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            avg_rating = df['rating'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{avg_rating:.2f}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Average Rating</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            critical_issues = len(df[df['rating'] <= 2])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{critical_issues}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Critical Issues</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            unique_centers = df['center_number'].nunique()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{unique_centers}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Service Centers</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Rating distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index()
            df_ratings = pd.DataFrame({
                'Rating': rating_counts.index,
                'Count': rating_counts.values
            })
            fig = px.bar(
                df_ratings,
                x='Rating',
                y='Count',
                labels={'Rating': 'Rating', 'Count': 'Count'},
                color='Rating',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                height=400,
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Feedback Trend")
            # Resample by day and count feedback
            df['date'] = df['created_at'].dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                labels={'date': 'Date', 'count': 'Number of Feedback'},
                markers=True
            )
            fig.update_layout(
                height=400,
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Recent feedback
        st.subheader("Recent Feedback")
        recent_feedback = df.sort_values('created_at', ascending=False).head(10)
        for _, row in recent_feedback.iterrows():
            with st.expander(f"Ref ID: {row['ref_id']} | Rating: {row['rating']} | Date: {row['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                cols = st.columns([2,1])
                with cols[0]:
                    st.write("**Comment:**")
                    st.write(row['comment'] if row['comment'] else "No comment provided")
                with cols[1]:
                    st.write("**Center:**")
                    st.write(row['center_number'] if row['center_number'] else "Unknown")
                    if row['document_url']:
                        st.write("**Document:**")
                        st.markdown(f"[View Document]({row['document_url']})")
    
    # Detailed Analytics
    elif selected == "Detailed Analytics":
        st.markdown('<p class="title-text">Detailed Analytics</p>', unsafe_allow_html=True)
        
        # Service center performance
        st.subheader("Service Center Performance")
        center_ratings = df.groupby('center_number')['rating'].agg(['mean', 'count']).reset_index()
        center_ratings.columns = ['Service Center', 'Average Rating', 'Feedback Count']
        center_ratings = center_ratings.sort_values('Average Rating', ascending=False)
        
        # Ensure we have data
        if not center_ratings.empty:
            fig = px.scatter(
                center_ratings,
                x='Feedback Count',
                y='Average Rating',
                size='Feedback Count',
                color='Average Rating',
                hover_name='Service Center',
                color_continuous_scale='RdYlGn',
                title='Service Center Performance'
            )
            fig.update_layout(
                height=500,
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(range=[0, 5.5])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            st.dataframe(center_ratings, use_container_width=True)
        
        # Rating distribution by day of week
        st.subheader("Rating by Day of Week")
        df['day_of_week'] = df['created_at'].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_ratings = df.groupby('day_of_week')['rating'].mean().reindex(day_order).reset_index()
        
        fig = px.bar(
            day_ratings,
            x='day_of_week',
            y='rating',
            labels={'day_of_week': 'Day of Week', 'rating': 'Average Rating'},
            color='rating',
            color_continuous_scale='RdYlGn',
        )
        fig.update_layout(
            height=400,
            margin=dict(t=30, b=10, l=10, r=10),
            yaxis=dict(range=[0, 5])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feedback Management
    elif selected == "Feedback Management":
        st.markdown('<p class="title-text">Feedback Management</p>', unsafe_allow_html=True)
        
        # Search and filter
        col1, col2 = st.columns([1, 1])
        with col1:
            search_term = st.text_input("Search by Ref ID, Contact or Center")
        with col2:
            rating_filter = st.multiselect("Filter by Rating", options=[1, 2, 3, 4, 5], default=None)
        
        # Apply filters
        filtered_df = df.copy()
        if search_term:
            filtered_df = filtered_df[
                filtered_df['ref_id'].str.contains(search_term, case=False, na=False) |
                filtered_df['citizen_contact'].str.contains(search_term, case=False, na=False) |
                filtered_df['center_number'].str.contains(search_term, case=False, na=False)
            ]
        
        if rating_filter:
            filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]
        
        # Display results in a table
        st.write(f"Showing {len(filtered_df)} results")
        display_df = filtered_df[['ref_id', 'citizen_contact', 'center_number', 'rating', 'comment', 'created_at']].copy()
        display_df.columns = ["Reference ID", "Contact", "Service Center", "Rating", "Comment", "Date & Time"]
        # Enhanced table with sorting
        st.dataframe(
            display_df,
            use_container_width=True
        )
        
        if 'sentiment' in df.columns:
            with st.expander("Sentiment Snapshot"):
                sent_counts = df['sentiment'].dropna().apply(lambda x: x.split(':')[0]).value_counts()
                st.bar_chart(sent_counts)
        if 'toxicity_score' in df.columns:
            with st.expander("Toxicity Distribution"):
                tox = df['toxicity_score'].dropna()
                if not tox.empty:
                    st.write(f"Avg toxicity: {tox.mean():.3f}")
                    fig = px.histogram(tox)
                    st.plotly_chart(fig)
        
        # Export options
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Export to CSV"):
                # Generate download link
                csv = filtered_df.to_csv(index=False)
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"feedback_export_{now}.csv",
                    mime="text/csv"
                )
    
    # Critical Issues
    elif selected == "Critical Issues":
        st.markdown('<p class="title-text">Critical Issues</p>', unsafe_allow_html=True)
        
        # Filter for critical issues (ratings <= 2)
        critical_df = df[df['rating'] <= 2].sort_values('created_at', ascending=False)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Critical Issues", len(critical_df))
        with col2:
            st.metric("Last 7 Days", len(critical_df[critical_df['created_at'] > (datetime.datetime.now() - timedelta(days=7))]))
        with col3:
            if not critical_df.empty:
                resolution_rate = "0%"  # In a real app, you might track resolution status
                st.metric("Resolution Rate", resolution_rate)
            else:
                st.metric("Resolution Rate", "N/A")
        
        # Display critical issues
        if not critical_df.empty:
            for _, row in critical_df.iterrows():
                with st.expander(f"⚠️ Critical Issue: {row['ref_id']} | Rating: {row['rating']} | Date: {row['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                    cols = st.columns([2,1])
                    with cols[0]:
                        st.write("**Contact:**")
                        st.write(row['citizen_contact'])
                        st.write("**Comment:**")
                        st.write(row['comment'] if row['comment'] else "No comment provided")
                    with cols[1]:
                        st.write("**Center:**")
                        st.write(row['center_number'] if row['center_number'] else "Unknown")
                        st.write("**Actions:**")
                        st.button("Mark Resolved", key=f"resolve_{row['id']}", disabled=True)
                        st.button("Send Follow-up", key=f"followup_{row['id']}", disabled=True)
        else:
            st.success("No critical issues found for the selected time period.")
    elif selected == "AI Insights":
        st.markdown('<p class="title-text">AI Insights Dashboard</p>', unsafe_allow_html=True)
        
        # Sentiment Analysis Section
        st.subheader("Sentiment Analysis")
        
        if 'sentiment' in df.columns:
            # Extract sentiment labels and scores
            df['sentiment_label'] = df['sentiment'].apply(
                lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else 'Unknown'
            )
            
            # Count sentiments
            sentiment_counts = df['sentiment_label'].value_counts()
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Sentiment distribution pie chart
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='Sentiment Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Sentiment trend over time
                df['date'] = pd.to_datetime(df['created_at']).dt.date
                sentiment_trend = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
                
                fig = px.line(
                    sentiment_trend,
                    x='date',
                    y='count',
                    color='sentiment_label',
                    title='Sentiment Trend Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sentiment analysis data not available. Please process feedback with AI services.")
        
        # Intent Classification Section
        st.subheader("User Intent Classification")

        if 'intent' in df.columns:
            # Count intents
            intent_counts = df['intent'].value_counts()
            
            # Create a dataframe for the intent counts
            df_intents = pd.DataFrame({
                'Intent': intent_counts.index,
                'Count': intent_counts.values
            })

            # Then use the dataframe in the chart
            fig = px.bar(
                df_intents,
                x='Intent',
                y='Count',
                title='Distribution of User Intents',
                labels={'Intent': 'Intent', 'Count': 'Count'},
                color='Intent'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Intent classification data not available.")
        
        # Language Distribution
        st.subheader("Language Distribution")
        
        if 'language' in df.columns:
            language_counts = df['language'].value_counts()
            
            fig = px.pie(
                values=language_counts.values,
                names=language_counts.index,
                title='Language Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Language detection data not available.")
        
        # Toxicity Analysis
        st.subheader("Content Toxicity Analysis")
        
        if 'toxicity_score' in df.columns:
            # Remove None values and convert to float
            toxicity_df = df[df['toxicity_score'].notna()].copy()
            toxicity_df['toxicity_score'] = toxicity_df['toxicity_score'].astype(float)
            
            # Define toxicity levels
            toxicity_df['toxicity_level'] = pd.cut(
                toxicity_df['toxicity_score'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Toxicity distribution
                toxicity_levels = toxicity_df['toxicity_level'].value_counts()
                
                fig = px.pie(
                    values=toxicity_levels.values,
                    names=toxicity_levels.index,
                    title='Toxicity Level Distribution',
                    color_discrete_sequence=['green', 'orange', 'red']
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Toxicity histogram
                fig = px.histogram(
                    toxicity_df,
                    x='toxicity_score',
                    nbins=10,
                    title='Toxicity Score Distribution',
                    color_discrete_sequence=['blue']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show high toxicity feedback
            if len(toxicity_df[toxicity_df['toxicity_level'] == 'High']) > 0:
                st.subheader("Potentially Problematic Feedback")
                toxic_df = toxicity_df[toxicity_df['toxicity_level'] == 'High'].sort_values('toxicity_score', ascending=False)
                
                for _, row in toxic_df.iterrows():
                    with st.expander(f"⚠️ High toxicity ({row['toxicity_score']:.2f}): {row['ref_id']}"):
                        st.write(f"**Comment:** {row['comment']}")
                        st.write(f"**Rating:** {row['rating']}/5")
                        st.write(f"**Center:** {row['center_number']}")
        else:
            st.info("Toxicity analysis data not available.")
            
        # Semantic Search (if embeddings are available)
        st.subheader("Semantic Search")
        
        if 'embedding' in df.columns and any(df['embedding'].notna()):
            search_query = st.text_input("Search feedback semantically (concepts, not just keywords):")
            
            if search_query:
                # Import required libraries for semantic search
                from sentence_transformers import SentenceTransformer
                import numpy as np
                
                @st.cache_resource
                def load_embedding_model():
                    return SentenceTransformer('all-MiniLM-L6-v2')
                
                model = load_embedding_model()
                
                # Generate embedding for search query
                query_embedding = model.encode(search_query)
                
                # Filter dataframe to only rows with embeddings
                df_with_embeddings = df[df['embedding'].notna()].copy()
                
                # Calculate similarity scores
                def calc_similarity(embedding_str):
                    if not embedding_str:
                        return 0
                    try:
                        emb = np.array([float(x) for x in embedding_str.split(',')])
                        return np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    except:
                        return 0
                
                df_with_embeddings['similarity'] = df_with_embeddings['embedding'].apply(calc_similarity)
                
                # Show top results
                results = df_with_embeddings.sort_values('similarity', ascending=False).head(5)
                
                st.write(f"Top {len(results)} semantically similar results:")
                
                for _, row in results.iterrows():
                    with st.expander(f"Similarity: {row['similarity']:.2f} - {row['comment'][:50]}..."):
                        st.write(f"**Full comment:** {row['comment']}")
                        st.write(f"**Rating:** {row['rating']}/5")
                        st.write(f"**Center:** {row['center_number']}")
                        st.write(f"**Date:** {row['created_at']}")
        else:
            st.info("Semantic search requires embedding data. Process feedback with AI services to enable this feature.")
        
        # Entity Extraction Visualization
        st.subheader("Entity Extraction")
        
        if 'entities' in df.columns and any(df['entities'].notna()):
            # Parse entities JSON strings
            entities_df = df[df['entities'].notna()].copy()
            
            def parse_entities(entities_json):
                try:
                    if isinstance(entities_json, str):
                        return json.loads(entities_json)
                    return {}
                except:
                    return {}
            
            entities_df['parsed_entities'] = entities_df['entities'].apply(parse_entities)
            
            # Extract specific entities
            service_centers = []
            amounts = []
            dates = []
            
            for _, row in entities_df.iterrows():
                entities = row['parsed_entities']
                
                if 'service_center' in entities and entities['service_center']:
                    service_centers.append(entities['service_center'])
                    
                if 'amount' in entities and entities['amount']:
                    amounts.append(entities['amount'])
                    
                if 'date' in entities and entities['date']:
                    dates.append(entities['date'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Service center distribution
                if service_centers:
                    service_center_counts = pd.Series(service_centers).value_counts().head(10)
                    
                    df_centers = pd.DataFrame({
                        'Service_Center': service_center_counts.index,
                        'Count': service_center_counts.values
                    })

                    fig = px.bar(
                        df_centers,
                        x='Service_Center',
                        y='Count',
                        title='Top Service Centers Mentioned',
                        labels={'Service_Center': 'Service Center', 'Count': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No service center entities extracted.")
            
            with col2:
                # Date distribution
                if dates:
                    date_counts = pd.Series(dates).value_counts().head(10)
                    
                    df_dates = pd.DataFrame({
                        'Date': date_counts.index,
                        'Count': date_counts.values
                    })
                    fig = px.bar(
                        df_dates,
                        x='Date',
                        y='Count',
                        title='Date Mentions in Feedback',
                        labels={'Date': 'Date', 'Count': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No date entities extracted.")
        else:
            st.info("Entity extraction data not available.")
    
    # Settings
    elif selected == "Settings":
        st.markdown('<p class="title-text">Dashboard Settings</p>', unsafe_allow_html=True)
        
        st.subheader("User Management")
        # In a real app, this would connect to your user database
        users = [
            {"username": "admin", "role": "Administrator", "last_login": "2023-07-01 10:30:45"},
            {"username": "manager", "role": "Manager", "last_login": "2023-06-29 15:22:10"},
        ]
        
        st.table(users)
        
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Database Status: Connected")
            st.info("WhatsApp API Status: Active")
        with col2:
            st.info("Last Backup: 2023-07-01 00:00:00")
            st.info("System Version: 1.0.0")
        
        if st.button("Run Database Cleanup"):
            count = database.cleanup_stale_conversations()
            st.success(f"Reset {count} stale conversations")
        
        if 'embedding' in df.columns and st.checkbox("Enable semantic search"):
            q = st.text_input("Semantic search query")
            if q:
                import numpy as np
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                qv = model.encode([q])[0]
                def parse_vec(v):
                    return np.array([float(x) for x in v.split(',')])
                df_local = df.dropna(subset=['embedding']).copy()
                df_local['sim'] = df_local['embedding'].apply(lambda v: float(np.dot(parse_vec(v), qv) /
                                                (np.linalg.norm(parse_vec(v))*np.linalg.norm(qv)+1e-9)))
                st.dataframe(df_local.sort_values('sim', ascending=False)[['ref_id','comment','rating','sim']].head(15))

# Main app execution
def main():
    # Move authentication check outside the render_dashboard call
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Show only login form when not authenticated
        st.markdown('<p class="title-text">Admin Dashboard Login</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.button("Login")
            
            if login_button:
                if username == "admin" and password == "admin123":
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.experimental_rerun()  # Try this first
                else:
                    st.error("Invalid credentials")
    else:
        # Only render dashboard when authenticated
        render_dashboard()

if __name__ == "__main__":
    main()