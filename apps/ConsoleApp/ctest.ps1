<#
.SYNOPSIS
	build.ps1 로 빌드한 뒤 테스트를 돌린다.

.DESCRIPTION
	decode  - CTC prefix beam search 단위 테스트
	samples - apps/cli/samples/<id>/ 를 인식해 파일명(정답)과 대조

.PARAMETER Config
	Release(기본) / Debug / RelWithDebInfo / MinSizeRel

.PARAMETER Clean
	build/ 를 지우고 처음부터.

.PARAMETER Filter
	테스트 이름 정규식 (ctest -R). 예: -Filter decode

.EXAMPLE
	.\ctest.ps1
	.\ctest.ps1 -Filter decode
#>
[CmdletBinding()]
param(
	[ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
	[string]$Config = 'Release',
	[switch]$Clean,
	[string]$Filter
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$buildDir = Join-Path $root 'build'

& (Join-Path $root 'build.ps1') -Config $Config -Clean:$Clean

# ctest.exe 로 명시한다. 이 스크립트와 이름이 같아 헷갈리지 않도록.
$ctestArgs = @('--test-dir', $buildDir, '-C', $Config, '--output-on-failure')
if ($Filter) { $ctestArgs += @('-R', $Filter) }

Write-Host ''
ctest.exe @ctestArgs
if ($LASTEXITCODE -ne 0) { throw "테스트 실패 (exit $LASTEXITCODE)" }
