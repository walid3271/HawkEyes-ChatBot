To Run:

git clone https://github.com/walid3271/Romero-ChatBot.git

pest your google API Key in the .env file (https://aistudio.google.com/app/apikey)

Open Command Prompt in Romero-ChatBot Directory

code . (to open the in vs-code)

Open the Command Prompt in the vs-code terminal

python -m venv venv (Create virtual environment)

.\venv\Scripts\activate.bat

pip install -r requirements.txt

cd ChatBot

streamlit run he.py (run interface)

Note: The response takes much time because of using a free API token and using only CPU power for similarity search.
