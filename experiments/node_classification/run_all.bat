@echo off
setlocal enabledelayedexpansion

REM Define datasets
set DATASETS=cora citeseer

REM Define splits
set SPLITS=MMF1 MMF2 MMF3

REM ---------------------------------------
REM First loop: save wavelets
REM ---------------------------------------
for %%D in (%DATASETS%) do (
    echo Generating wavelets for dataset=%%D
    python main.py ^
        --dataset %%D ^
        --dim 2 ^
        --seed 42 ^
        --output-dir .\%%D_wavelets ^
        --save-wavelets
)

REM ---------------------------------------
REM Second loop: run splits
REM ---------------------------------------
for %%D in (%DATASETS%) do (
    for %%S in (%SPLITS%) do (
        REM Convert split to lowercase manually
        set SPLIT_LOWER=%%S
        if /I "%%S"=="MMF1" set SPLIT_LOWER=mmf1
        if /I "%%S"=="MMF2" set SPLIT_LOWER=mmf2
        if /I "%%S"=="MMF3" set SPLIT_LOWER=mmf3

        echo Running dataset=%%D split=%%S
        python main.py ^
            --dataset %%D ^
            --dim 2 ^
            --split %%S ^
            --seed 42 ^
            --output-dir .\%%D_!SPLIT_LOWER! ^
            --load-wavelets .\%%D_wavelets\wavelets
    )
)

echo All runs completed.
pause