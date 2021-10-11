@echo off
set /p imgdims="Enter tile dimensions (px in one dimension): "
py C:\Users\Manuel\Nextcloud\Masterarbeit\py3\DataPreparation.py -imgd %imgdims%
echo "Tiles created."
pause
