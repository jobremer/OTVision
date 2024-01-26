echo Install OTVision.
@REM @echo off
@REM FOR /F "tokens=* USEBACKQ" %%F IN (`python --version`) DO SET PYTHON_VERSION=%%F

@REM echo %PYTHON_VERSION%
@REM if "x%PYTHON_VERSION:3.10=%"=="x%PYTHON_VERSION%" (
@REM     echo "Python Version 3.10 is not installed in environment." & cmd /K & exit
@REM )

C:\Python310\python.exe -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir%
deactivate
