@echo off
REM Batch script for running learnable_mmf_basis

set program=learnable_mmf_basis
set data_folder=..\..\data\

REM Dataset
set dataset=MUTAG

REM Other datasets. Remember to change the MMF hyperparameters accordingly to the size of each dataset.
REM set dataset=PTC
REM set dataset=DD
REM set dataset=NCI1

set dir=%program%

REM Create directories
if not exist "%dir%" mkdir "%dir%"
cd "%dir%"
if not exist "%dataset%" mkdir "%dataset%"
cd ..

REM Parameters
set K=2
set drop=1
set dim=2
set epochs=1024
set learning_rate=1e-3

set name=%dataset%.%program%

REM Run the Python script
python %program%.py --data_folder=%data_folder% --dir=%dir% --dataset=%dataset% --name=%name% --K=%K% --drop=%drop% --dim=%dim% --epochs=%epochs% --learning_rate=%learning_rate%

pause