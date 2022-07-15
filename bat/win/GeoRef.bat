@echo off
set /p b="Select block: "
set /p p="Select plot: "
set in="F:\Block_%b%\Block%b%_%p%\Block%b%_%p%"
set out="F:\Block_%b%\Block%b%_%p%GeoTIFF"
@echo Input folder: %in%
@echo Started: %date% %time%
py C:\Users\Manuel\Nextcloud\Masterarbeit\py3\fun\ApproxGeoRef.py %in% %out%
@echo Completed: %date% %time%
pause