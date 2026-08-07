<#
.SYNOPSIS
	CMake configure. Visual Studio 번들 CMake를 찾아 PATH에 얹고 build/ 를 만든다.

.DESCRIPTION
	cmake가 PATH에 없으면 vswhere로 Visual Studio를 찾아 번들 CMake 경로를 이 프로세스의
	PATH에 추가한다(전역 PATH는 건드리지 않는다). build.ps1이 이 스크립트를 호출한다.

.PARAMETER Clean
	build/ 를 지우고 처음부터 configure. 제너레이터를 바꿨을 때 필요하다.

.EXAMPLE
	.\config.ps1
	.\config.ps1 -Clean
#>
[CmdletBinding()]
param(
	[switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$buildDir = Join-Path $root 'build'

function Add-CMakeToPath {
	if (Get-Command cmake -ErrorAction SilentlyContinue) { return }

	$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
	if (Test-Path $vswhere) {
		# 설치가 진행 중이면 -latest 가 비어 나오므로 -all 로 한 번 더 본다.
		$paths = @(& $vswhere -latest -products * -property installationPath) +
		         @(& $vswhere -all -products * -property installationPath)
		foreach ($vs in ($paths | Where-Object { $_ })) {
			$bin = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
			if (Test-Path (Join-Path $bin 'cmake.exe')) {
				$env:PATH = "$bin;$env:PATH"
				return
			}
		}
	}
	throw "cmake를 찾을 수 없습니다. 'winget install Kitware.CMake' 로 설치하거나, Visual Studio Build Tools에서 'C++를 사용한 데스크톱 개발' 워크로드를 설치하세요."
}

Add-CMakeToPath
Write-Host "cmake : $((Get-Command cmake).Source)"

if ($Clean -and (Test-Path $buildDir)) {
	Write-Host "clean : $buildDir 삭제"
	Remove-Item -Recurse -Force $buildDir
}

# 제너레이터는 지정하지 않는다 — CMake가 설치된 최신 Visual Studio를 고르므로
# VS를 업그레이드해도 이 스크립트를 고칠 필요가 없다.
cmake -S $root -B $buildDir -A x64
if ($LASTEXITCODE -ne 0) {
	Write-Host ''
	Write-Host 'configure 실패. 자주 있는 원인:' -ForegroundColor Yellow
	Write-Host '  - VS 설치가 아직 안 끝남 : vswhere.exe -all -products * -format json 에서 isComplete 확인'
	Write-Host '  - 제너레이터를 바꾼 경우  : .\config.ps1 -Clean 으로 build/ 를 지우고 재실행'
	Write-Host '  - 최초 configure 는 ONNX Runtime(~120MB)을 받으므로 네트워크가 필요합니다'
	throw "cmake configure 실패 (exit $LASTEXITCODE)"
}
