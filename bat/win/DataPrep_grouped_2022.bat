set /p imgdims="Enter tile dimensions (px in one dimension): "
set /p dname="Save to (directory): "
@echo Started: %date% %time%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\DataPreparation.py -name %dname% -imgd %imgdims% -grp -abc -1 -wd home -date "02_2022"
@echo Completed: %date% %time%
pause