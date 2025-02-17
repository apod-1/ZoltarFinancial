#!/bin/bash

# Activate the appropriate virtual environment if necessary
# source /path/to/your/venv/bin/activate

# Start both Streamlit apps
streamlit run app/Strategy_Play.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &
streamlit run CustomizeMyCV/app/CustomizeMyCV.py --server.port 8502 --server.enableCORS false --server.enableXsrfProtection false &

wait