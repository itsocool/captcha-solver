@echo off
for /f "delims=" %%i in ('main.exe -i=".\_internal\captcha_data\supreme_court\0\images\draft\스크린샷 5.JPG"') do set "result=%%i"
echo %result%