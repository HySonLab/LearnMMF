@echo off
REM ==========================================
REM Single Experiment Runner for Windows
REM ==========================================

REM Initialize and activate conda environment
echo Initializing Anaconda environment...

REM Try common Anaconda locations
IF EXIST "%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Anaconda3\Scripts\activate.bat"
) ELSE IF EXIST "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) ELSE IF EXIST "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\Anaconda3\Scripts\activate.bat"
)

echo Activating LearnMMF environment...
call conda activate LearnMMF
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate LearnMMF environment
    echo.
    echo Please ensure:
    echo   1. Anaconda/Miniconda is installed
    echo   2. LearnMMF environment exists
    echo.
    echo To create the environment:
    echo   conda create -n LearnMMF python=3.8
    echo   conda activate LearnMMF
    echo   pip install torch numpy tqdm
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Running Single Experiment
echo ==========================================
echo.

REM Run the experiment with default or provided arguments
python main.py %*

REM Check if successful
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Experiment failed
    echo.
) ELSE (
    echo.
    echo ==========================================
    echo Experiment completed successfully!
    echo ==========================================
    echo.
)

REM Deactivate environment
call conda deactivate

pause
exit /b %ERRORLEVEL%