# 배포용 zip 생성: 실행 파일 + DLL + 모델 + 샘플 + 문서
#
# 사용법: powershell -ExecutionPolicy Bypass -File tools\package.ps1
# 산출물: dist\captcha-cli-<version>-windows-x86_64.zip
$ErrorActionPreference = 'Stop'

Set-Location (Join-Path $PSScriptRoot '..')

$version = (Select-String -Path Cargo.toml -Pattern '^version *= *"(.*)"').Matches[0].Groups[1].Value
$name = "captcha-cli-$version-windows-x86_64"
$stage = "dist\$name"

if (-not (Test-Path target\release\captcha-cli.exe)) {
    Write-Error "먼저 cargo build --release 를 실행하세요"
}

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null

# ONNX Runtime 프리빌트가 함께 떨군 DLL(DirectML 등)도 exe 옆에 담는다
Copy-Item target\release\captcha-cli.exe, target\release\*.dll $stage
Copy-Item README.md, pred.sh, pred.cmd, pred.ps1 $stage
Copy-Item models, samples $stage -Recurse

$zip = "dist\$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip
Remove-Item $stage -Recurse -Force

"$zip  ({0:N0} MB)" -f ((Get-Item $zip).Length / 1MB)
