set omk_dir=G:\Masterarbeit\dat\test_Monitoring512_21
set out_dir=G:\Masterarbeit\dat\test_Monitoring512_21
set mod_dir=G:\Masterarbeit\out\
set mod_subdir=\cpts\trained_mod
set mod_name=Monitoring
set /p epoch="Enter best epoch: "
set /p gpu_use="Use GPU (y/n): "
IF "%gpu_use%"=="y" (
    set ugpu= -gpu
) ELSE (
    set ugpu=
)
set mod_path=%mod_dir%%mod_name%%mod_subdir%\Epoch.%epoch%.hdf5
@echo Started: %date% %time%
set tstart=%time%
set omk_name=ortho_B3_2_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%\%omk_name%.tif
set out_path=%out_dir%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set omk_dir=G:\Masterarbeit\dat\test_Monitoring512_22
set out_dir=G:\Masterarbeit\dat\test_Monitoring512_22
set omk_name=ortho_B1_1_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%\%omk_name%.tif
set out_path=%out_dir%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set omk_name=ortho_B2_3_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%\%omk_name%.tif
set out_path=%out_dir%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set omk_name=ortho_B3_2_CROP
set out_name=%omk_name:CROP=PRED%
set omk_path=%omk_dir%\%omk_name%.tif
set out_path=%out_dir%\%out_name%.tif
@echo Input: %omk_path%
@echo Output: %out_path%
@echo Model: %mod_path%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\LeleNet_prd.py %omk_path% %out_path% %mod_path% -rol 0.25 -col 0.25%ugpu%
set tend=%time%
@echo Started %tstart%, completed %tend% on %date%.
pause