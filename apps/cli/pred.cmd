@echo off
REM Launcher for pred.ps1 -- see that file for what this actually does.
REM
REM Kept ASCII on purpose. cmd.exe reads batch files through the console codepage,
REM so Korean text in a .cmd is mojibake on CP949/CP437, and `chcp 65001` is not a
REM fix: switching codepage mid-file desyncs cmd's parser, and switching back from
REM 65001 to 949 wipes the console buffer. PowerShell writes Unicode to the console
REM directly and has none of these problems, so all output lives in pred.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pred.ps1" %*
