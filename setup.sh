#!/bin/bash

# setup.sh - Streamlit Cloud setup script

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml
