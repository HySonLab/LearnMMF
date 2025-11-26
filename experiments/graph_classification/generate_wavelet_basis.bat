@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Parametric Experiment Runner with Timing
rem Usage: generate_wavelet_basis.bat [dataset] [method]
rem Examples:
rem   generate_wavelet_basis.bat MUTAG baseline
rem   generate_wavelet_basis.bat PTC evolutionary_algorithm
rem   generate_wavelet_basis.bat DD directed_evolution
rem   generate_wavelet_basis.bat NCI1 k_neighbours
rem ============================================================================

rem === Parse command line arguments ===
set DATASET=%1
set METHOD=%2

rem === Validate arguments ===
if "%DATASET%"=="" (
    echo Error: Dataset not specified
    echo.
    echo Usage: generate_wavelet_basis.bat [dataset] [method]
    echo.
    echo Available datasets:
    echo   MUTAG, PTC, DD, NCI1
    echo.
    echo Available methods:
    echo   baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution
    echo.
    echo Example: generate_wavelet_basis.bat MUTAG baseline
    goto end
)

if "%METHOD%"=="" (
    echo Error: Method not specified
    echo.
    echo Usage: generate_wavelet_basis.bat [dataset] [method]
    echo.
    echo Available methods:
    echo   baseline              - Baseline MMF
    echo   random                - Random MMF
    echo   k_neighbours          - k-Neighbors MMF
    echo   evolutionary_algorithm- Evolutionary Algorithm MMF
    echo   directed_evolution    - Directed Evolution MMF
    echo.
    echo Example: generate_wavelet_basis.bat MUTAG baseline
    goto end
)

rem === Configuration ===
set DATA_FOLDER=..\..\data\
set CONDA_ENV=LearnMMF

rem === Activate conda environment ===
echo Activating conda environment: %CONDA_ENV%...
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo Error: Failed to activate conda environment '%CONDA_ENV%'
    echo Please ensure conda is initialized and the environment exists.
    echo You can create it with: conda create -n LearnMMF python=3.8
    goto end
)
echo Conda environment '%CONDA_ENV%' activated successfully.
echo.

rem === Set dataset-specific parameters ===
if /i "%DATASET%"=="MUTAG" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
) else if /i "%DATASET%"=="PTC" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
) else if /i "%DATASET%"=="DD" (
    set DIM=150
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
) else if /i "%DATASET%"=="NCI1" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
) else (
    echo Error: Unknown dataset '%DATASET%'
    echo.
    echo Available datasets: MUTAG, PTC, DD, NCI1
    goto end
)

rem === Create timing log directory ===
if not exist "timing_logs" mkdir "timing_logs"
set TIMING_LOG=timing_logs\timing_summary.txt

rem === Display configuration ===
echo.
echo ========================================
echo Wavelet Basis Generation with Timing
echo ========================================
echo Dataset: %DATASET%
echo Method:  %METHOD%
echo ========================================
echo.

rem === Record start time ===
call :get_timestamp START_TIME

rem === Route to appropriate method ===
if /i "%METHOD%"=="baseline" goto run_baseline
if /i "%METHOD%"=="random" goto run_random
if /i "%METHOD%"=="k_neighbours" goto run_k_neighbours
if /i "%METHOD%"=="evolutionary_algorithm" goto run_ea
if /i "%METHOD%"=="directed_evolution" goto run_de

echo Error: Unknown method '%METHOD%'
echo.
echo Available methods:
echo   baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution
goto end

rem ============================================================================
:run_baseline
rem ============================================================================
echo Running Baseline MMF on %DATASET%...
echo Start time: %START_TIME%
echo.

set PROGRAM=baseline_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --dim=%DIM% ^
    --seed=42

call :log_completion "Baseline MMF"
goto end

rem ============================================================================
:run_random
rem ============================================================================
echo Running Random MMF on %DATASET%...
echo Start time: %START_TIME%
echo.

set PROGRAM=random_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --K=%K% ^
    --drop=%DROP% ^
    --dim=%DIM% ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

call :log_completion "Random MMF"
goto end

rem ============================================================================
:run_k_neighbours
rem ============================================================================
echo Running k-Neighbors MMF on %DATASET%...
echo Start time: %START_TIME%
echo.

set PROGRAM=k_neighbours_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --K=%K% ^
    --drop=%DROP% ^
    --dim=%DIM% ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

call :log_completion "k-Neighbors MMF"
goto end

rem ============================================================================
:run_ea
rem ============================================================================
echo Running Evolutionary Algorithm MMF on %DATASET%...
echo Start time: %START_TIME%
echo.

set PROGRAM=metaheuristics_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --K=%K% ^
    --dim=%DIM% ^
    --method=ea ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

call :log_completion "Evolutionary Algorithm MMF"
goto end

rem ============================================================================
:run_de
rem ============================================================================
echo Running Directed Evolution MMF on %DATASET%...
echo Start time: %START_TIME%
echo.

set PROGRAM=metaheuristics_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --K=%K% ^
    --dim=%DIM% ^
    --method=de ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

call :log_completion "Directed Evolution MMF"
goto end

rem ============================================================================
rem Helper Functions
rem ============================================================================

:setup_directories
set DIR=%~1
if not exist "%DIR%" mkdir "%DIR%"
pushd "%DIR%"
if not exist "%DATASET%" mkdir "%DATASET%"
popd
exit /b

:get_timestamp
rem Get current timestamp
set %1=%date% %time%
exit /b

:calculate_duration
rem Calculate duration between START_TIME and current time
rem This is a simplified version - shows start and end times
call :get_timestamp END_TIME
exit /b

:log_completion
set METHOD_NAME=%~1
call :calculate_duration

echo.
echo ========================================
echo %METHOD_NAME% completed for %DATASET%
echo ========================================
echo Start time:  %START_TIME%
echo End time:    %END_TIME%
echo Results:     %METHOD%\%DATASET%\
echo ========================================

rem === Log to timing summary file ===
echo [%date% %time%] %DATASET% - %METHOD_NAME% >> %TIMING_LOG%
echo   Start:  %START_TIME% >> %TIMING_LOG%
echo   End:    %END_TIME% >> %TIMING_LOG%
echo   Output: %METHOD%\%DATASET%\ >> %TIMING_LOG%
echo. >> %TIMING_LOG%

rem === Also log to method-specific log ===
set METHOD_LOG=%METHOD%\%DATASET%\timing.log
echo Wavelet Basis Generation Timing >> %METHOD_LOG%
echo ================================== >> %METHOD_LOG%
echo Dataset:    %DATASET% >> %METHOD_LOG%
echo Method:     %METHOD_NAME% >> %METHOD_LOG%
echo Start time: %START_TIME% >> %METHOD_LOG%
echo End time:   %END_TIME% >> %METHOD_LOG%
echo ================================== >> %METHOD_LOG%

echo.
echo Timing information saved to:
echo   - %TIMING_LOG%
echo   - %METHOD_LOG%

exit /b

rem ============================================================================
:end
rem ============================================================================
echo.
echo ========================================
echo Execution Complete
echo ========================================
echo.
echo View timing summary: type timing_logs\timing_summary.txt

rem === Deactivate conda environment ===
call conda deactivate

endlocal