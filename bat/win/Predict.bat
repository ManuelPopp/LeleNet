set omk_dir=G:\Masterarbeit\dat\test_
set out_dir=G:\Masterarbeit\dat\test_
set mod_dir=G:\Masterarbeit\Models\
set mod_subdir=\cpts\trained_mod
set /p omk_name="Enter orthomosaic filename: "
set /p out_name="Enter output filename: "
set /p mod_name="Enter model name: "
set /p gpu_use="Use GPU (y/n): "
IF "%gpu_use%"=="y" (
    set ugpu= -gpu
) ELSE (
    set ugpu=
)
set omk_path=%omk_dir%%mod_name%\%omk_name%.tif
set out_path=%out_dir%%mod_name%\%out_name%.tif
set mod_path=%mod_dir%%mod_name%%mod_subdir%
@echo Started: %date% %time%
set tstart=%time%
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%

set tend=%time%
@echo Started %tstart%, completed %tend% on %date%.
pause
