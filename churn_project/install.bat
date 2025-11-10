\
@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✅ Environment ready. To activate later: call .venv\Scripts\activate
