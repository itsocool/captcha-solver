@echo off
xcopy .\dist\main c:\appz\main /E /I /Y
cd c:\appz\main
main.exe %*