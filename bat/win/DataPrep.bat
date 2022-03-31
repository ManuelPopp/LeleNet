set /p imgdims="Enter tile dimensions (px in one dimension): "
@echo Started: %date% %time%
py C:\Users\Manuel\Nextcloud\Masterarbeit\py3\DataPreparation.py -imgd %imgdims%
@echo Completed: %date% %time%
pause
