@echo off
setlocal enabledelayedexpansion

rem === Configuration ===
set program=train_wavelet_network
set data_folder=..\..\data\

rem === Dataset (choose one) ===
set dataset=MUTAG
rem set dataset=PTC
rem set dataset=DD
rem set dataset=NCI1

rem === Wavelet basis type ===
set wavelet_type=baseline_mmf_basis
rem set wavelet_type=learnable_mmf_basis

rem === Create directories ===
if not exist %program% mkdir %program%
cd %program%
if not exist %dataset% mkdir %dataset%
cd ..
set dir=%program%\%dataset%

rem === File paths ===
set adjs=%wavelet_type%\%dataset%\%dataset%.%wavelet_type%.adjs.pt
set laplacians=%wavelet_type%\%dataset%\%dataset%.%wavelet_type%.laplacians.pt
set mother_wavelets=%wavelet_type%\%dataset%\%dataset%.%wavelet_type%.mother_wavelets.pt
set father_wavelets=%wavelet_type%\%dataset%\%dataset%.%wavelet_type%.father_wavelets.pt

rem === Hyperparameters ===
set num_epoch=256
set num_layers=6
set hidden_dim=32

rem === Cross-validation training ===
for %%s in (0 1 2 3 4 5 6 7 8 9) do (
    set name=%program%.dataset.%dataset%.split.%%s.num_epoch.%num_epoch%.num_layers.%num_layers%.hidden_dim.%hidden_dim%
    echo Running split %%s ...
    python %program%.py --dataset=%dataset% --data_folder=%data_folder% --dir=%dir% --name=!name! --num_epoch=%num_epoch% --adjs=%adjs% --laplacians=%laplacians% --mother_wavelets=%mother_wavelets% --father_wavelets=%father_wavelets% --split=%%s --num_layers=%num_layers% --hidden_dim=%hidden_dim%
)

rem === Summary of results ===
echo.
echo ===== Summary of Results =====
findstr "Best accuracy:" %program%\%dataset%\*.log

endlocal
