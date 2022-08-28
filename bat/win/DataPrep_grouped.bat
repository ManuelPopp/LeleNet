set /p imgdims="Enter tile dimensions (px in one dimension): "
set /p dname="Enter dataset main folder name: "
@echo Started: %date% %time%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\DataPreparation.py -name %dname% -imgd %imgdims% -grp -abc -1 -wd home
@echo Completed: %date% %time%
pause