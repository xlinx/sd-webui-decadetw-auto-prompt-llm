@echo off
echo. 2>%~dp0running.txt
set COUNT=3
for /l %%a in (1,1,1) do call :run
goto :EOF
:run
set /a rand=%random% %% %COUNT%
for /f "tokens=1* delims=:" %%i in ('findstr /n .* "C:\Users\x\_ML\stable-diffusion-webui\extensions\sd-webui-decadetw-auto-prompt-llm\before-after-actions\read_random_line.bat"') do (
    if "%%i"=="%rand%" echo %%j
)