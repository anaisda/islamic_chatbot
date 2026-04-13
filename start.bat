@echo off
echo.
echo  Islamic Library Chatbot
echo  ========================

if not exist ".env" (
    echo  .env introuvable — creation depuis .env.example
    copy .env.example .env
    echo  Ouvre .env et configure ta cle OpenAI
    pause
    exit
)

echo  Demarrage du serveur...
echo  Ouvre http://localhost:5000 dans ton navigateur
echo.

python app.py
pause
