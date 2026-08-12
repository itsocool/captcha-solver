<#
.SYNOPSIS
	CMake configure 후 빌드한다. cmake 가 PATH 에 없으면 Visual Studio 번들 CMake 를 찾아 얹는다.

.DESCRIPTION
	cmake 가 PATH 에 없으면 vswhere 로 Visual Studio 를 찾아 번들 CMake 경로를 이 프로세스의
	PATH 에 추가한다(전역 PATH 는 건드리지 않는다). 담을 캡차 유형은 config.json 이 정한다.

.PARAMETER Config
	Release(기본) / Debug / RelWithDebInfo / MinSizeRel

.PARAMETER Clean
	build/ 를 지우고 처음부터. 제너레이터를 바꿨을 때 필요하다.

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
	Write-Host '  - 제너레이터를 바꾼 경우  : .\build.ps1 -Clean 으로 build/ 를 지우고 재실행'
	Write-Host '  - config.json 의 캡차 유형이 captcha_data 에 없음'
	Write-Host '  - 최초 configure 는 ONNX Runtime(~120MB)을 받으므로 네트워크가 필요합니다'
	throw "cmake configure 실패 (exit $LASTEXITCODE)"
}

cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "빌드 실패 (exit $LASTEXITCODE)" }

Write-Host ''
# Join-Path 는 Windows PowerShell 5.1 에서 인자를 2개만 받는다.
Write-Host "완료 : $(Join-Path (Join-Path $buildDir $Config) 'captcha.exe')" -ForegroundColor Green
