<#
.SYNOPSIS
	build.ps1 로 빌드한 뒤 NSIS 설치 파일을 만든다.

.PARAMETER Config
	Release(기본) / Debug / RelWithDebInfo / MinSizeRel

.PARAMETER Clean
	build/ 를 지우고 처음부터. config.ps1 로 그대로 전달된다.

.EXAMPLE
	.\pack.ps1
	.\pack.ps1 -Config RelWithDebInfo
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

# cpack 이 makensis 를 레지스트리로도 찾지만, 없을 때 나오는 메시지가 불친절해서 먼저 확인한다.
if (-not (Get-Command makensis -ErrorAction SilentlyContinue)) {
	$nsis = @("${env:ProgramFiles(x86)}\NSIS", "$env:ProgramFiles\NSIS") |
		Where-Object { Test-Path (Join-Path $_ 'makensis.exe') } | Select-Object -First 1
	if (-not $nsis) {
		throw "makensis 를 찾을 수 없습니다. 'winget install NSIS.NSIS' 로 설치하세요."
	}
	$env:PATH = "$nsis;$env:PATH"
}
Write-Host "makensis : $((Get-Command makensis).Source)"

# 빌드. cmake 를 PATH 에 얹는 것도 여기서 끝나므로 cpack 도 바로 쓸 수 있다.
& (Join-Path $root 'build.ps1') -Config $Config -Clean:$Clean

cpack --config (Join-Path $buildDir 'CPackConfig.cmake') -C $Config -B $buildDir
if ($LASTEXITCODE -ne 0) { throw "cpack 실패 (exit $LASTEXITCODE)" }

Write-Host ''
Get-ChildItem (Join-Path $buildDir 'captcha-solver-*.exe') |
	ForEach-Object { Write-Host "완료 : $($_.FullName)  ($([math]::Round($_.Length / 1MB, 1)) MB)" -ForegroundColor Green }
