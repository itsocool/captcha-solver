@echo off
for /f "delims=" %%i in ('main.exe -c=gov24 -i=".\_internal\captcha_data\gov24\1\images\draft\원본 5.png" -v') do set "result=%%i"
echo %result%