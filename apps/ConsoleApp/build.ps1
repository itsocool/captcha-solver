<#
.SYNOPSIS
	config.ps1 로 configure한 뒤 빌드한다.

.PARAMETER Config
	Release(기본) / Debug / RelWithDebInfo / MinSizeRel

.PARAMETER Clean
	build/ 를 지우고 처음부터. config.ps1 로 그대로 전달된다.

.EXAMPLE
	.\build.ps1
	.\build.ps1 -Config Debug
	.\build.ps1 -Clean
#>
[CmdletBinding()]
param(
	[ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
	[string]$Config = 'Release',
	[switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$buildDir = Join-Path $root 'build'

# configure. cmake 를 PATH 에 얹는 것도 여기서 한다(프로세스 환경이라 이 뒤로도 유효).
& (Join-Path $root 'config.ps1') -Clean:$Clean

cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "빌드 실패 (exit $LASTEXITCODE)" }

Write-Host ''
# Join-Path 는 Windows PowerShell 5.1 에서 인자를 2개만 받는다.
Write-Host "완료 : $(Join-Path (Join-Path $buildDir $Config) 'captcha.exe')" -ForegroundColor Green
