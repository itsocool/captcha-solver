@echo off
REM 대법원(supreme_court) 샘플 이미지를 CLI로 모두 인식하고 정답(파일명)과 대조한다.
REM 사용법: pred.cmd [captcha_id]
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "CAPTCHA_ID=%~1"
if "%CAPTCHA_ID%"=="" set "CAPTCHA_ID=supreme_court"

set "SAMPLES_DIR=samples\%CAPTCHA_ID%"
REM 배포 번들에서는 스크립트 옆에, 개발 중에는 target\release\에 바이너리가 있다
set "CLI=captcha-cli.exe"
if not exist "%CLI%" set "CLI=target\release\captcha-cli.exe"

if not exist "%CLI%" (
	echo CLI 바이너리가 없습니다: %CLI%  ^(cargo build --release^)
	exit /b 2
)
if not exist "%SAMPLES_DIR%" (
	echo 샘플 디렉터리가 없습니다: %SAMPLES_DIR%
	exit /b 2
)

echo 이미지              정답        예측        결과
echo --------------------------------------------------------------

set /a total=0
set /a match=0

for %%F in ("%SAMPLES_DIR%\*.png") do (
	set "NAME=%%~nxF"
	set "TRUTH=%%~nF"

	for /f "usebackq delims=" %%P in (`"%CLI%" -c %CAPTCHA_ID% -i "%%F"`) do set "PRED=%%P"

	set /a total+=1
	if "!PRED!"=="!TRUTH!" (
		set /a match+=1
		set "MARK=O"
	) else (
		set "MARK=X"
	)

	echo !NAME!    !TRUTH!    !PRED!    !MARK!
)

echo --------------------------------------------------------------
if %total%==0 (
	echo 샘플 이미지가 없습니다: %SAMPLES_DIR%
	exit /b 2
)
echo %CAPTCHA_ID%: %match%/%total% 일치

if not %match%==%total% exit /b 1
exit /b 0
