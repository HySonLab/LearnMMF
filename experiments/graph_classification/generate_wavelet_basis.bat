@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Parametric Experiment Runner
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
    echo   baseline, random, k_neighbours, metaheuristics
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
    echo   random          - Random MMF
    echo   k_neighbours    - k-Neighbors MMF
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
) else if /i "%DATASET%"=="PTC" (
    set DIM=2
    set K=2
    set DROP=1
    set EPOCHS=1024
    set LEARNING_RATE=1e-3
) else if /i "%DATASET%"=="DD" (
    set DIM=2
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
if /i "%METHOD%"=="random" goto run_random
if /i "%METHOD%"=="k_neighbours" goto run_k_neighbours
if /i "%METHOD%"=="evolutionary_algorithm" goto run_ea
if /i "%METHOD%"=="directed_evolution" goto run_de

echo Error: Unknown method '%METHOD%'
echo.
echo Available methods:
echo   baseline, random, evolutionary_algorithm, directed_evolution, k_neighbours
goto end

rem ============================================================================
:run_baseline
rem ============================================================================
echo Running Baseline MMF on %DATASET%...
echo.

set PROGRAM=baseline_mmf_basis
call :setup_directories %METHOD%

set NAME=%DATASET%.%METHOD%

python %PROGRAM%.py ^
    --data_folder=%DATA_FOLDER% ^
    --dir=%METHOD% ^
    --dataset=%DATASET% ^
    --name=%NAME% ^
    --dim=%DIM%
    --seed=42

echo.
echo Baseline MMF completed for %DATASET%
echo Results: %METHOD%\%DATASET%\
goto end

rem ============================================================================
:run_random
rem ============================================================================
echo Running random MMF on %DATASET%...
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

echo.
echo Random MMF completed for %DATASET%
echo Results: %METHOD%\%DATASET%\
goto end

rem ============================================================================
:run_k_neighbours
rem ============================================================================
echo Running k Neighbors MMF on %DATASET%...
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

echo.
echo Random MMF completed for %DATASET%
echo Results: %METHOD%\%DATASET%\
goto end

rem ============================================================================
:run_ea
rem ============================================================================
echo Running evolutionary algorithm MMF on %DATASET%...
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
    --drop=%DROP% ^
    --dim=%DIM% ^
    --method=ea ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

echo.
echo Random MMF completed for %DATASET%
echo Results: %METHOD%\%DATASET%\
goto end

rem ============================================================================
:run_de
rem ============================================================================
echo Running directed evolution MMF on %DATASET%...
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
    --drop=%DROP% ^
    --dim=%DIM% ^
    --method=de ^
    --epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --seed=42

echo.
echo Random MMF completed for %DATASET%
echo Results: %METHOD%\%DATASET%\
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