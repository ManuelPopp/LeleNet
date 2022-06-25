set omk_dir=G:\Masterarbeit\dat\test_1024_21\
set out_dir=G:\Masterarbeit\dat\test_1024_21\
set mod_dir=C:\Users\Manuel\Dropbox\Models\
set mod_subdir=\cpts\trained_mod
set /p omk_name="Enter orthomosaic filename: "
set /p out_name="Enter output filename: "
set /p mod_name="Enter model name: "
set omk_path=%omk_dir%%omk_name%
set out_path=%out_dir%%out_name%
set mod_path=%mod_dir%%mod_name%%mod_subdir%
@echo Started: %date% %time%
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25
@echo Completed: %date% %time%
pause