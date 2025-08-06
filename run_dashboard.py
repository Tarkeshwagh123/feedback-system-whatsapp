import os
import subprocess

def run_streamlit():
    """Run the Streamlit dashboard"""
    try:
        print("Starting Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "streamlit_dashboard.py"], check=True)
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit()