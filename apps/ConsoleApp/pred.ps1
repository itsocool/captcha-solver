# 샘플 이미지를 captcha.exe로 인식하고 정답(파일명)과 대조한다.
# apps/cli/pred.ps1 의 C++ 버전. 기본은 captcha_data 에서 100장을 무작위로 뽑는다.
#
# 사용법: powershell -ExecutionPolicy Bypass -File pred.ps1 [captcha_id] [-Count N] [-Seed N]
#
# 이 파일은 UTF-8 BOM 으로 저장해야 한다. BOM 이 없으면 Windows PowerShell 5.1 이
# ANSI(CP949 등)로 읽어 아래 한글이 깨진다. PowerShell 은 콘솔에 유니코드로 직접
# 쓰기 때문에 chcp 를 건드릴 필요가 없다.
param(
    [string]$CaptchaId = 'supreme_court',
    [int]$Count = 100,
    [int]$Seed,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release'
)

Set-Location $PSScriptRoot

# 배포 번들에서는 스크립트 옆에, 개발 중에는 build\<Config>\ 에 실행 파일이 있다
$exe = '.\captcha.exe'
if (-not (Test-Path $exe)) { $exe = ".\build\$Config\captcha.exe" }
if (-not (Test-Path $exe)) {
    Write-Host "실행 파일이 없습니다: $exe (.\build.ps1 -Config $Config)"
    exit 2
}

# 이미지 출처: captcha_data 의 pred 세트를 우선 쓰고, 없으면 cli 샘플 10장으로 폴백한다.
# rev 는 메타데이터에 적힌 것을 쓰고, 메타가 없으면 가장 큰 번호를 고른다.
function Resolve-ImageDir {
    $meta = "..\cli\models\$CaptchaId.meta.json"
    $rev = $null
    if (Test-Path $meta) { $rev = (Get-Content $meta -Raw | ConvertFrom-Json).rev }

    $base = "..\..\captcha_data\$CaptchaId"
    if (Test-Path $base) {
        if ($null -ne $rev -and (Test-Path "$base\$rev\images\pred")) { return "$base\$rev\images\pred" }
        $newest = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^\d+$' } |
            Sort-Object { [int]$_.Name } -Descending |
            Select-Object -First 1
        if ($newest -and (Test-Path "$($newest.FullName)\images\pred")) { return "$($newest.FullName)\images\pred" }
    }

    foreach ($dir in @("samples\$CaptchaId", "..\cli\samples\$CaptchaId")) {
        if (Test-Path $dir) { return $dir }
    }
    return $null
}

$imageDir = Resolve-ImageDir
if (-not $imageDir) {
    Write-Host "이미지 디렉터리를 찾을 수 없습니다: captcha_data\$CaptchaId 또는 ..\cli\samples\$CaptchaId"
    exit 2
}

$all = @(Get-ChildItem "$imageDir\*.png" -File)
if ($all.Count -eq 0) {
    Write-Host "이미지가 없습니다: $imageDir"
    exit 2
}

# Get-Random -Count 는 전체보다 크게 주면 그냥 전부(섞어서) 돌려준다.
if ($PSBoundParameters.ContainsKey('Seed')) { Get-Random -SetSeed $Seed | Out-Null }
$images = @(Get-Random -InputObject $all -Count ([Math]::Min($Count, $all.Count)))

Write-Host ("출처 : {0}  ({1}장 중 {2}장 무작위)" -f (Resolve-Path $imageDir).Path, $all.Count, $images.Count)
Write-Host ''

# captcha.exe 는 --json 이 없어 예측 문자열만 내보낸다. 신뢰도 열이 없는 대신
# 시간은 프로세스 전체 실행 시간이다(ONNX 세션 초기화 포함이라 Rust CLI 의
# 내부 측정값보다 크게 나온다).
$rowFormat = '{0,-16} {1,-10} {2,-10} {3,-8} {4}'

# 헤더는 -f 패딩을 쓰지 않는다. 한글은 화면에서 2칸을 차지하는데 -16 같은 패딩은
# 글자 수로 세기 때문에 데이터 행과 어긋난다. 데이터 행의 열 위치(17/28/39/48)에
# 맞춰 공백을 직접 넣었다.
'이미지           정답       예측       시간     결과'
'------------------------------------------------------'

$total = 0
$match = 0
$swTotal = [Diagnostics.Stopwatch]::StartNew()

foreach ($image in $images) {
    $truth = $image.BaseName
    $total++

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $out = & $exe "-c=$CaptchaId" "-i=$($image.FullName)" 2>&1
    $sw.Stop()
    $elapsed = '{0} ms' -f [int]$sw.Elapsed.TotalMilliseconds

    if ($LASTEXITCODE -ne 0) {
        ($rowFormat -f $image.Name, $truth, '-', $elapsed, '❌') +
            '  ' + ($out | Select-Object -First 1)
        continue
    }

    if ($out -eq $truth) { $match++; $mark = '✅' } else { $mark = '❌' }
    $rowFormat -f $image.Name, $truth, $out, $elapsed, $mark
}

$swTotal.Stop()
'------------------------------------------------------'
# 헤더와 같은 이유로 라벨 뒤 공백을 직접 맞췄다(값은 모두 12번째 칸에서 시작).
'캡차        {0}' -f $CaptchaId
'일치        {0}/{1}' -f $match, $total
'일치율      {0:P1}' -f ($match / $total)
'총 소요     {0:N1}초  (평균 {1} ms)' -f $swTotal.Elapsed.TotalSeconds,
    [int]($swTotal.Elapsed.TotalMilliseconds / $total)

if ($match -ne $total) { exit 1 }
exit 0
