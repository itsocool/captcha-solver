# 대법원(supreme_court) 샘플 이미지를 CLI로 모두 인식하고 정답(파일명)과 대조한다.
#
# 사용법: powershell -ExecutionPolicy Bypass -File pred.ps1 [captcha_id]
#
# 이 파일은 UTF-8 BOM 으로 저장해야 한다. BOM 이 없으면 Windows PowerShell 5.1 이
# ANSI(CP949 등)로 읽어 아래 한글이 깨진다. PowerShell 은 콘솔에 유니코드로 직접
# 쓰기 때문에 pred.cmd 와 달리 chcp 를 건드릴 필요가 없다.
param([string]$CaptchaId = 'supreme_court')

Set-Location $PSScriptRoot

$samplesDir = "samples\$CaptchaId"
# 배포 번들에서는 스크립트 옆에, 개발 중에는 target\release\에 바이너리가 있다
$cli = '.\captcha-cli.exe'
if (-not (Test-Path $cli)) { $cli = '.\target\release\captcha-cli.exe' }

if (-not (Test-Path $cli)) {
    Write-Host "CLI 바이너리가 없습니다: $cli (cargo build --release)"
    exit 2
}
if (-not (Test-Path $samplesDir)) {
    Write-Host "샘플 디렉터리가 없습니다: $samplesDir"
    exit 2
}

'{0,-16} {1,-10} {2,-10} {3,-10} {4}' -f '이미지', '정답', '예측', '신뢰도', '시간'
'--------------------------------------------------------------'

$total = 0
$match = 0

foreach ($image in Get-ChildItem "$samplesDir\*.png" -File) {
    $truth = $image.BaseName
    $total++

    $out = & $cli -c $CaptchaId -i $image.FullName --json 2>&1
    if ($LASTEXITCODE -ne 0) {
        '{0,-16} {1,-10} 추론 실패: {2}' -f $image.Name, $truth, ($out | Select-Object -First 1)
        continue
    }

    $r = $out | ConvertFrom-Json
    if ($r.prediction -eq $truth) { $match++; $mark = 'O' } else { $mark = 'X' }
    '{0,-16} {1,-10} {2,-10} {3,-10:F4} {4} ms  {5}' -f `
        $image.Name, $truth, $r.prediction, $r.confidence, $r.elapsed_ms, $mark
}

'--------------------------------------------------------------'
if ($total -eq 0) {
    Write-Host "샘플 이미지가 없습니다: $samplesDir"
    exit 2
}
"{0}: {1}/{2} 일치" -f $CaptchaId, $match, $total

if ($match -ne $total) { exit 1 }
exit 0
