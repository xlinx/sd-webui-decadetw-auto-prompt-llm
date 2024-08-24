
@REM echo. 2>%~dp0running.txt
set COUNT=3
for /l %%a in (1,1,1) do call :run
goto :EOF
:run
set /a rand=%random% %% %COUNT%
for /f "tokens=1* delims=:" %%i in ('findstr /n .* "read_random_line.bat"') do (
    if "%%i"=="%rand%" echo %%j
)