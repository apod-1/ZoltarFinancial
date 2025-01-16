#!/bin/bash

# Ensure we're using the correct Python version
PYTHON_VERSION=$(python --version)

if [[ "$PYTHON_VERSION" == *"3.12"* ]]; then
    echo "Switching to Python 3.11 for compatibility."
    # Activate a virtual environment with Python 3.11 if necessary
    source /path/to/python3.11/venv/bin/activate
fi

streamlit run app/Strategy_Play.py --server.port 8501 &
streamlit run CustomizeMyCV/app/CustomizeMyCV.py --server.port 8502 &

wait