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
            ["Overview", "Detailed Analytics", "Feedback Management", "Critical Issues", "Settings"],
            icons=["house", "graph-up", "chat-left-text", "exclamation-triangle", "gear"],
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
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                color=rating_counts.index,
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
        
        # Enhanced table with sorting
        st.dataframe(
            filtered_df[['ref_id', 'citizen_contact', 'center_number', 'rating', 'comment', 'created_at']],
            use_container_width=True,
            column_config={
                "ref_id": "Reference ID",
                "citizen_contact": "Contact",
                "center_number": "Service Center",
                "rating": "Rating",
                "comment": "Comment",
                "created_at": "Date & Time"
            },
            hide_index=True
        )
        
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

# Main app execution
def main():
    if authenticate():
        render_dashboard()

if __name__ == "__main__":
    main()