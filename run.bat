@echo off
cd C:\Projekt\marketscreener
call C:\Projekt\marketscreener\investing_env\Scripts\activate.bat
start "" cmd /c "streamlit run Marketscreener.py"
exit