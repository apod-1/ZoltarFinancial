#!/bin/bash

streamlit run app/Strategy_Play.py --server.port 8501 &
streamlit run CustomizeMyCV/app/CustomizeMyCV.py --server.port 8502 &

wait