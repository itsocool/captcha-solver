<#
.SYNOPSIS
	build.ps1 로 빌드한 뒤 NSIS 설치 파일을 만들고 packages\cli\ 에 놓는다.

.DESCRIPTION
	%ProgramFiles%\captcha-cli\bin\ 에 실행 파일 + ONNX Runtime DLL + models/ 를 설치하고,
	설치 마법사가 bin 을 PATH 에 추가할지 묻는다. 제어판에서 제거할 수 있다.
	담을 캡차 유형은 config.json 이 정한다(build.ps1 이 처리).
	.nsi 는 dist\ 에 임시로 만들어 쓰므로 따로 관리할 파일이 없다.

.PARAMETER Clean
	빌드 전 cargo clean. build.ps1 로 그대로 전달된다.

.EXAMPLE
	.\pack.ps1
#>
[CmdletBinding()]
param(
	[switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# cpack 처럼 레지스트리를 뒤지지 않으므로 여기서 직접 찾는다.
if (-not (Get-Command makensis -ErrorAction SilentlyContinue)) {
	$nsis = @("${env:ProgramFiles(x86)}\NSIS", "$env:ProgramFiles\NSIS") |
		Where-Object { Test-Path (Join-Path $_ 'makensis.exe') } | Select-Object -First 1
	if (-not $nsis) {
		throw "makensis 를 찾을 수 없습니다. 'winget install NSIS.NSIS' 로 설치하세요."
	}
	$env:PATH = "$nsis;$env:PATH"
}
Write-Host "makensis : $((Get-Command makensis).Source)"

& (Join-Path $PSScriptRoot 'build.ps1') -Clean:$Clean

$version = (Select-String -Path Cargo.toml -Pattern '^version *= *"(.*)"').Matches[0].Groups[1].Value
$release = 'target\release'
$stage = 'dist\stage'
$setup = "dist\captcha-cli-$version-win64-setup.exe"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null

# ONNX Runtime 프리빌트가 함께 떨군 DLL(DirectML 등)도 exe 옆에 담는다.
Copy-Item "$release\captcha-cli.exe", "$release\*.dll" $stage -ErrorAction SilentlyContinue
Copy-Item README.md $stage
Copy-Item "$release\models" $stage -Recurse

$icon = Resolve-Path '..\..\hi.works.ico' -ErrorAction SilentlyContinue
$iconLine = if ($icon) { "Icon `"$($icon.Path)`"`nUninstallIcon `"$($icon.Path)`"" } else { '' }

# NSIS 는 $INSTDIR 같은 자기 변수를 쓰므로, PowerShell 이 건드리지 않게 리터럴 here-string 이다.
# PATH 편집은 PowerShell 에 맡긴다 — NSIS 문자열은 1024자 제한이 있어 긴 PATH 를 잘라먹는다.
# ($$ 는 NSIS 에서 리터럴 $ 로 풀려 PowerShell 변수가 된다)
$nsi = @'
Unicode true
Name "captcha-cli {VERSION}"
OutFile "{OUTFILE}"
InstallDir "$PROGRAMFILES64\captcha-cli"
InstallDirRegKey HKLM "Software\captcha-cli" "InstallDir"
RequestExecutionLevel admin
ShowInstDetails show
{ICON}

!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\captcha-cli"
!define PS "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"

Page components
Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "captcha-cli (필수)" SecMain
	SectionIn RO
	SetOutPath "$INSTDIR\bin"
	File /r "{STAGE}\*.*"

	WriteUninstaller "$INSTDIR\uninstall.exe"
	WriteRegStr HKLM "Software\captcha-cli" "InstallDir" "$INSTDIR"
	WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "captcha-cli"
	WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "{VERSION}"
	WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "HiWorks"
	WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
	WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
	WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
	WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "PATH 에 bin 추가" SecPath
	DetailPrint "PATH 에 $INSTDIR\bin 추가"
	nsExec::ExecToLog '"${PS}" -NoProfile -ExecutionPolicy Bypass -Command "$$d=$\"$INSTDIR\bin$\"; $$p=[Environment]::GetEnvironmentVariable($\"Path$\",$\"Machine$\"); if(($$p -split $\";$\") -notcontains $$d){[Environment]::SetEnvironmentVariable($\"Path$\",($$p.TrimEnd($\";$\")+$\";$\"+$$d),$\"Machine$\")}"'
	Pop $0
SectionEnd

Section "uninstall"
	nsExec::ExecToLog '"${PS}" -NoProfile -ExecutionPolicy Bypass -Command "$$d=$\"$INSTDIR\bin$\"; $$p=[Environment]::GetEnvironmentVariable($\"Path$\",$\"Machine$\"); [Environment]::SetEnvironmentVariable($\"Path$\",((($$p -split $\";$\") | Where-Object {$$_ -and $$_ -ne $$d}) -join $\";$\"),$\"Machine$\")"'
	Pop $0

	; 설치한 것만 지운다. RMDir /r "$INSTDIR" 는 사용자가 경로를 바꿨을 때 위험하다.
	RMDir /r "$INSTDIR\bin"
	Delete "$INSTDIR\uninstall.exe"
	RMDir "$INSTDIR"

	DeleteRegKey HKLM "${UNINST_KEY}"
	DeleteRegKey HKLM "Software\captcha-cli"
SectionEnd
'@

$nsi = $nsi.Replace('{VERSION}', $version).
	Replace('{OUTFILE}', (Join-Path $PSScriptRoot $setup)).
	Replace('{STAGE}', (Join-Path $PSScriptRoot $stage)).
	Replace('{ICON}', $iconLine)

# makensis 는 BOM 이 없으면 ACP 로 읽어 한글 주석에서 죽는다. PS 5.1/7 양쪽에서 BOM 을 보장한다.
$nsiPath = 'dist\captcha-cli.nsi'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot $nsiPath), $nsi, [Text.UTF8Encoding]::new($true))

makensis $nsiPath
if ($LASTEXITCODE -ne 0) { throw "makensis 실패 (exit $LASTEXITCODE)" }

Remove-Item $stage -Recurse -Force

# 설치 파일과 그 설치본에 담긴 캡차 목록을 함께 배치한다.
$packages = Join-Path $PSScriptRoot '..\..\packages\cli'
New-Item -ItemType Directory -Force $packages | Out-Null
Copy-Item $setup $packages -Force
Copy-Item 'config.json' $packages -Force

Write-Host ''
Write-Host "완료 : $(Resolve-Path (Join-Path $packages (Split-Path $setup -Leaf)))  ($([math]::Round((Get-Item $setup).Length / 1MB, 1)) MB)" -ForegroundColor Green
