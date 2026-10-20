@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_HOME%\bin;%PATH%"
"E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe" "E:\Develop\Deepseek\tinyinfer\prototypes\kernels\test_official_kernels.py" %*
