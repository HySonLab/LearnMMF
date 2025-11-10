@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Parametric Experiment Runner
rem Usage: generate_wavelet_basis.bat [dataset] [method]
rem Examples:
rem   generate_wavelet_basis.bat MUTAG baseline
rem   generate_wavelet_basis.bat PTC learnable
rem   generate_wavelet_basis.bat DD metaheuristics
rem   generate_wavelet_basis.bat NCI1 metaheuristics
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
    echo   baseline, learnable, metaheuristics
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
    echo   baseline        - Baseline MMF
    echo   learnable       - Learnable MMF
    echo   metaheuristics  - Metaheuristics MMF
    echo.
    echo Example: generate_wavelet_basis.bat MUTAG baseline
    goto end
)

rem === Configuration ===
set DATA_FOLDER=..\..\data\

rem === Set dataset-specific parameters ===
if /i "%DATASET%"=="MUTAG" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
    set NUM_LAYERS=6
    set HIDDEN_DIM=32
) else if /i "%DATASET%"=="PTC" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
    set NUM_LAYERS=6
    set HIDDEN_DIM=32
) else if /i "%DATASET%"=="DD" (
    set DIM=3
    set K=3
    set DROP=1
    set EPOCHS=2048
    set LEARNING_RATE=5e-4
    set NUM_LAYERS=8
    set HIDDEN_DIM=64
) else if /i "%DATASET%"=="NCI1" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
    set NUM_LAYERS=6
    set HIDDEN_DIM=32
) else (
    echo Error: Unknown dataset '%DATASET%'
    echo.
    echo Available datasets: MUTAG, PTC, DD, NCI1
    goto end
)

rem === Display configuration ===
echo.
echo ========================================
echo Parametric Experiment Runner
echo ========================================
echo Dataset: %DATASET%
echo Method:  %METHOD%
echo ========================================
echo.

rem === Route to appropriate method ===
if /i "%METHOD%"=="baseline" goto run_baseline
if /i "%METHOD%"=="learnable" goto run_learnable
if /i "%METHOD%"=="metaheuristics" goto run_metaheuristics

echo Error: Unknown method '%METHOD%'
echo.
echo Available methods:
echo   baseline, learnable, metaheuristics
goto end

rem ============================================================================
:run_baseline
rem ============================================================================
echo Running Baseline MMF on %DATASET%...
echo.

set PROGRAM=baseline_mmf_basis
call :setup_directories %PROGRAM%

set NAME=%DATASET%.%PROGRAM%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%PROGRAM% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --dim=%DIM%

echo.
echo Baseline MMF completed for %DATASET%
echo Results: %PROGRAM%\%DATASET%\
goto end

rem ============================================================================
:run_learnable
rem ============================================================================
echo Running Learnable MMF on %DATASET%...
echo.

set PROGRAM=learnable_mmf_basis
call :setup_directories %PROGRAM%

set NAME=%DATASET%.%PROGRAM%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%PROGRAM% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --K=%K% ^
    --drop=%DROP% ^
    --dim=%DIM% ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE%

echo.
echo Learnable MMF completed for %DATASET%
echo Results: %PROGRAM%\%DATASET%\
goto end

rem ============================================================================
:run_metaheuristics
rem ============================================================================
echo Running Metaheuristics MMF on %DATASET%...
echo.

set PROGRAM=metaheuristics_mmf_basis
call :setup_directories %PROGRAM%

set NAME=%DATASET%.%PROGRAM%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%PROGRAM% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --dim=%DIM%

echo.
echo Metaheuristics MMF completed for %DATASET%
echo Results: %PROGRAM%\%DATASET%\
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

rem ============================================================================
:end
rem ============================================================================
echo.
echo ========================================
echo Execution Complete
echo ========================================
endlocal