@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Wavelet Neural Network Trainer
rem Usage: train_wavelet_network.bat [dataset] [method]
rem Examples:
rem   train_wavelet_network.bat MUTAG baseline
rem   train_wavelet_network.bat PTC random
rem   train_wavelet_network.bat DD k_neighbours
rem   train_wavelet_network.bat NCI1 evolutionary_algorithm
rem ============================================================================

rem === Parse command line arguments ===
set DATASET=%1
set METHOD=%2

rem === Validate arguments ===
if "%DATASET%"=="" (
    echo Error: Dataset not specified.
    echo Usage: train_wavelet_network.bat [dataset] [method]
    echo.
    echo Available datasets: MUTAG, PTC, DD, NCI1
    goto end
)

if "%METHOD%"=="" (
    echo Error: Method not specified.
    echo Usage: train_wavelet_network.bat [dataset] [method]
    echo.
    echo Available methods: baseline, random, k_neighbours, evolutionary_algorithm, directed_evolution
    goto end
)

rem === Configuration ===
set PROGRAM=train_wavelet_network
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

rem === Hyperparameters ===
set NUM_EPOCH=256
set NUM_LAYERS=6
set HIDDEN_DIM=32

rem === Paths ===
set BASIS_DIR=%METHOD%\%DATASET%
set OUTPUT_DIR=%PROGRAM%\%METHOD%\%DATASET%

rem === Check if basis files exist ===
if not exist "%BASIS_DIR%" (
    echo Error: Wavelet basis not found for method '%METHOD%' and dataset '%DATASET%'.
    echo Expected directory: %BASIS_DIR%
    echo Please run generate_wavelet_basis.bat first.
    goto end
)

rem === Create output directories ===
if not exist "%PROGRAM%" mkdir "%PROGRAM%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

rem === Basis file paths ===
set ADJS=%BASIS_DIR%\%DATASET%.%METHOD%.adjs.pt
set LAPLACIANS=%BASIS_DIR%\%DATASET%.%METHOD%.laplacians.pt
set MOTHER_WAVELETS=%BASIS_DIR%\%DATASET%.%METHOD%.mother_wavelets.pt
set FATHER_WAVELETS=%BASIS_DIR%\%DATASET%.%METHOD%.father_wavelets.pt

rem === Verify all required files exist ===
for %%F in ("%ADJS%" "%LAPLACIANS%" "%MOTHER_WAVELETS%" "%FATHER_WAVELETS%") do (
    if not exist %%F (
        echo Error: Missing required file: %%F
        goto end
    )
)

rem === Display configuration summary ===
echo.
echo ========================================
echo Wavelet Neural Network Training
echo ========================================
echo Dataset:      %DATASET%
echo Method:       %METHOD%
echo Output Dir:   %OUTPUT_DIR%
echo ----------------------------------------
echo Num Epochs:   %NUM_EPOCH%
echo Num Layers:   %NUM_LAYERS%
echo Hidden Dim:   %HIDDEN_DIM%
echo ========================================
echo.

rem === Cross-validation training ===
for %%S in (0 1 2 3 4 5 6 7 8 9) do (
    set NAME=%PROGRAM%.dataset.%DATASET%.method.%METHOD%.split.%%S.num_epoch.%NUM_EPOCH%.num_layers.%NUM_LAYERS%.hidden_dim.%HIDDEN_DIM%
    echo Running split %%S ...
    python %PROGRAM%.py ^
        --dataset=%DATASET% ^
        --data_folder=%DATA_FOLDER% ^
        --dir=%OUTPUT_DIR% ^
        --name=!NAME! ^
        --num_epoch=%NUM_EPOCH% ^
        --adjs=%ADJS% ^
        --laplacians=%LAPLACIANS% ^
        --mother_wavelets=%MOTHER_WAVELETS% ^
        --father_wavelets=%FATHER_WAVELETS% ^
        --split=%%S ^
        --num_layers=%NUM_LAYERS% ^
        --hidden_dim=%HIDDEN_DIM%
)

rem === Summary of results ===
echo.
echo ========================================
echo Summary of Results
echo ========================================
echo.

rem === Extract best accuracies and save to temporary file ===
set RESULTS=%OUTPUT_DIR%\accuracies.txt
if exist "%RESULTS%" del "%RESULTS%"

echo Extracting best accuracies from logs...
for %%S in (0 1 2 3 4 5 6 7 8 9) do (
    set NAME=%PROGRAM%.dataset.%DATASET%.method.%METHOD%.split.%%S.num_epoch.%NUM_EPOCH%.num_layers.%NUM_LAYERS%.hidden_dim.%HIDDEN_DIM%
    set LOGFILE=%OUTPUT_DIR%\!NAME!.log
    if exist "!LOGFILE!" (
        for /f "tokens=3" %%A in ('findstr "Best accuracy:" "!LOGFILE!"') do (
            echo %%A >> "%RESULTS%"
        )
    )
)

rem === Display individual results ===
echo Individual Results:
echo -------------------
set /a SPLIT_NUM=0
for /f %%A in (%RESULTS%) do (
    echo Split !SPLIT_NUM!: %%A
    set /a SPLIT_NUM+=1
)
echo.

:end
echo.
echo ========================================
echo Execution Complete
echo ========================================

rem === Deactivate conda environment ===
call conda deactivate

endlocal