@echo off
setlocal enabledelayedexpansion

rem === Configuration ===
set program=baseline_mmf_basis
set data_folder=..\..\data\

rem === Dataset (choose one) ===
set dataset=MUTAG
rem set dataset=PTC
rem set dataset=DD
rem set dataset=NCI1

rem === Create directories ===
set dir=%program%
if not exist %dir% mkdir %dir%
cd %dir%

if not exist %dataset% mkdir %dataset%
cd ..

rem === Parameters ===
set dim=2
set name=%dataset%.%program%

rem === Run Python script ===
python %program%.py --data_folder=%data_folder% --dir=%dir% --dataset=%dataset% --name=%name% --dim=%dim%

endlocal
