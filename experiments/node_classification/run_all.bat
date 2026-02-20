@echo off

set DATASETS=cora citeseer
set SPLITS=MMF1 MMF2 MMF3

for %%D in (%DATASETS%) do (
    for %%S in (%SPLITS%) do (
        echo Running dataset=%%D split=%%S

        set OUTPUT_DIR=%%D_%%S
        call set OUTPUT_DIR=%%D_%%S
        call set OUTPUT_DIR=%%D_%%S

        python main.py ^
            --dataset %%D ^
            --dim 2 ^
            --split %%S ^
            --seed 42 ^
            --output-dir .\%%D_%%S

        echo.
    )
)

echo All runs completed.
pause