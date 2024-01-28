python -m venv venv-listen
source venv-listen/bin/activate
pip install --upgrade -r ./requirements.txt
uvicorn app.main:app
