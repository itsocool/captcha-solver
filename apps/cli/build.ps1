<#
.SYNOPSIS
	captcha-cli 를 빌드하고 config.json 이 정한 캡차 모델을 실행 파일 옆으로 복사한다.

.PARAMETER Config
	Release(기본) / Debug

.PARAMETER Clean
	target/ 을 지우고 처음부터.

.EXAMPLE
	.\build.ps1
	.\build.ps1 -Config Debug
	.\build.ps1 -Clean
#>
[CmdletBinding()]
param(
	[ValidateSet('Release', 'Debug')]
	[string]$Config = 'Release',
	[switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
	throw "cargo 를 찾을 수 없습니다. 'winget install Rustlang.Rustup' 으로 설치하세요."
}

if ($Clean) { cargo clean }

# Debug 는 --release 없이 빌드한다. target 하위 디렉터리 이름도 그대로 따라간다.
$profileDir = $Config.ToLower()
if ($Config -eq 'Release') { cargo build --release } else { cargo build }
if ($LASTEXITCODE -ne 0) { throw "빌드 실패 (exit $LASTEXITCODE)" }

$outDir = Join-Path 'target' $profileDir
$exe = Join-Path $outDir 'captcha-cli.exe'

# 담을 캡차 유형은 config.json 의 "captchas" 배열이 정한다. 파일이 없으면 supreme_court 만.
$configFile = Join-Path $PSScriptRoot 'config.json'
if (Test-Path $configFile) {
	$wanted = @((Get-Content $configFile -Raw | ConvertFrom-Json).captchas)
	if ($wanted.Count -eq 0) { throw "캡차 유형 목록이 비어 있습니다: $configFile" }
	Write-Host "캡차 유형 목록($configFile): $($wanted -join ', ')"
} else {
	$wanted = @('supreme_court')
	Write-Host "$configFile 없음 — 기본값 supreme_court 만 담습니다." -ForegroundColor Yellow
}

# 산출물 디렉터리를 통째로 옮겨도 돌도록 모델을 exe 옆에 둔다.
# 목록에서 뺀 유형이 남지 않도록 매번 새로 만든다.
$modelsDir = Join-Path $outDir 'models'
if (Test-Path $modelsDir) { Remove-Item $modelsDir -Recurse -Force }
New-Item -ItemType Directory -Force $modelsDir | Out-Null

foreach ($id in $wanted) {
	$model = "..\..\models\$id.ort"
	$meta = "..\..\models\$id.meta.json"
	if (-not (Test-Path $model) -or -not (Test-Path $meta)) {
		# 유형이 빠진 배포본이 조용히 나가는 것보다 여기서 멈추는 편이 낫다.
		throw "'$id' 의 모델이 없습니다: $model / $meta`n(저장소 루트에서 'uv run python apps/cli/tools/sync_models.py' 를 실행하세요)"
	}
	Copy-Item $model, $meta $modelsDir -Force
}

Write-Host ''
Write-Host "완료 : $(Resolve-Path $exe)" -ForegroundColor Green
Write-Host "모델 : $($wanted.Count)종 ($($wanted -join ', '))"
