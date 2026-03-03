#!/bin/bash
echo "installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
python main_ec.py -l -a 0.1
python main_ec.py -l -a 0.01
python main_ec.py -l -a 0.001
python main_ec.py -l -a 0.001
python main_ec.py -l -a 0.0001
