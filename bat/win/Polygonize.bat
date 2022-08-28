set /p var_pd="Enter directory containing the prediction raster data: "
@echo Started: %date% %time%
set tstart=%time%
py D:\Dateien\Studium_KIT\Master_GOEK\Masterarbeit\py3\PolygonizePredictions.py %var_pd%
set tend=%time%
@echo Started %tstart%, completed %tend% on %date%.
pause