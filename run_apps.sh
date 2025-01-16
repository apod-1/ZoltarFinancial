#!/bin/bash

# Activate the appropriate virtual environment if necessary
#source /path/to/your/venv/bin/activate

# Start both Streamlit apps
streamlit run app/Strategy_Play.py --server.port 8501 &
streamlit run CustomizeMyCV/app/CustomizeMyCV.py --server.port 8502 &

wait