set /p omk_dir="Enter directory containing the orthomosaics: "
set /p out_dir="Enter output directory: "
set /p mod_path="Enter full path to the trained model: "
set /p gpu_use="Use GPU (y/n): "
IF "%gpu_use%"=="y" (
    set ugpu= -gpu
) ELSE (
    set ugpu=
)
@echo Started: %date% %time%
set tstart=%time%
cd /D "%omk_dir%"
SETLOCAL ENABLEDELAYEDEXPANSION
for %%i in (*) do (
	set out_name="%%i"
        call set out_name=%%out_name:.tif=_PRED.tif%%
	call set omk_path=%%omk_dir%%\%%i
	call set out_path=%%out_dir%%\%%out_name:"=%%%
	@echo Input: !omk_path!
	@echo Output: !out_path!
	@echo Model: !mod_path!
	py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py !omk_path! !out_path! %mod_path% -rol 0.25 -col 0.25%ugpu%
)
set tend=%time%
@echo Started %tstart%, completed %tend% on %date%.
pause
