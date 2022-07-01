set omk_dir=G:\Masterarbeit\dat\test_
set out_dir=G:\Masterarbeit\dat\test_
set mod_dir=G:\Masterarbeit\Models\
set mod_subdir=\cpts\trained_mod
set /p mod_name="Enter model name: "
set /p mod_path="Enter full model path: "
set /p gpu_use="Use GPU (y/n): "
IF "%gpu_use%"=="y" (
    set ugpu= -gpu
) ELSE (
    set ugpu=
)
@echo Started: %date% %time%
set tstart=%time%
set omk_name=ortho_B4_1_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%%mod_name%\%omk_name%.tif
set out_path=%out_dir%%mod_name%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set omk_name=ortho_B6_2_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%%mod_name%\%omk_name%.tif
set out_path=%out_dir%%mod_name%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set omk_name=ortho_B7_5_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%%mod_name%\%omk_name%.tif
set out_path=%out_dir%%mod_name%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set tend=%time%
@echo Started %tstart%, completed %tend% on %date%.
pause