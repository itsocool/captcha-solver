<#
    captchaSolver Spring Boot 개발서버 관리 (server.sh 의 PowerShell 포트)

        .\server.ps1 start|stop|restart|status

    application.yml 의 모델·DB 경로가 상대경로(../../models 등)라서 이 스크립트는
    항상 자기 위치(apps/springBoot)로 이동한 뒤 maven 을 띄운다.

    Windows 에는 POSIX 프로세스 그룹이 없다. 대신 mvn -> java 자식까지 통째로 정리하기
    위해 taskkill /T (프로세스 트리 종료) 를 쓴다. graceful 종료 신호(SIGTERM)에 대응하는
    것이 없어 stop 은 taskkill /T(그레이스풀 시도) 후 안 죽으면 /T /F(강제)로 넘어간다.

    환경변수
      PORT           서버 포트 (기본 5000)
      START_TIMEOUT  기동 대기 한계, 초 (기본 120)
      STOP_TIMEOUT   graceful 종료 대기 한계, 초 (기본 20)
      MVN            maven 실행 파일 (기본 mvn)

    PowerShell 7+ 필요 (?? 연산자, Get-NetTCPConnection).
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'help')]
    [string]$Command = 'help'
)

$ErrorActionPreference = 'Stop'

$AppDir   = $PSScriptRoot
$StateDir = Join-Path $AppDir '.dev'
$PidFile  = Join-Path $StateDir 'server.pid'
$LogFile  = Join-Path $StateDir 'server.log'

$Port         = [int]($env:PORT          ?? 5000)
$StartTimeout = [int]($env:START_TIMEOUT ?? 120)
$StopTimeout  = [int]($env:STOP_TIMEOUT  ?? 20)
$Mvn          = $env:MVN ?? 'mvn'
$HealthUrl    = "http://127.0.0.1:$Port/"

function Write-Log  { param([string]$Message) Write-Host $Message }
function Write-Warn { param([string]$Message) Write-Host $Message -ForegroundColor Yellow }
function Die {
    param([string]$Message)
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 상태 조회

# 저장된 PID. 없거나 유효하지 않으면 $null.
function Read-ServerPid {
    if (-not (Test-Path $PidFile)) { return $null }
    $raw = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($raw -match '^\d+$' -and [int]$raw -gt 1) { return [int]$raw }
    return $null
}

function Test-ProcessAlive {
    param([Nullable[int]]$ProcessId)
    if (-not $ProcessId) { return $false }
    return [bool](Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

# PORT 를 LISTEN 중인 프로세스 PID 들.
function Get-PortListenerPids {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
}

function Test-PortInUse {
    return [bool](Get-PortListenerPids)
}

function Get-PortListenerList {
    return (Get-PortListenerPids) -join ' '
}

function Test-HealthOk {
    try {
        Invoke-WebRequest -Uri $HealthUrl -TimeoutSec 3 -UseBasicParsing | Out-Null
        return $true
    } catch {
        return $false
    }
}

# 죽은 프로세스가 남긴 PID 파일 정리.
function Clear-StalePid {
    $p = Read-ServerPid
    if ($p -and -not (Test-ProcessAlive $p)) {
        Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
    }
}

function Show-LogTail {
    param([int]$Lines = 30)
    if ((Test-Path $LogFile) -and (Get-Item $LogFile).Length -gt 0) {
        Write-Warn "--- $LogFile (마지막 $Lines 줄) ---"
        Get-Content $LogFile -Tail $Lines | ForEach-Object { Write-Warn $_ }
        Write-Warn "--- 로그 끝 ---"
    }
}

# 프로세스 트리를 종료하고 사라질 때까지 기다린다. Force 없으면 graceful 시도.
function Stop-ProcessTreeAndWait {
    param([int]$ProcessId, [switch]$Force, [int]$Timeout)

    $args = @('/PID', $ProcessId, '/T')
    if ($Force) { $args += '/F' }
    & taskkill @args *> $null

    $waited = 0
    while ($waited -lt $Timeout) {
        if (-not (Test-ProcessAlive $ProcessId)) { return $true }
        Start-Sleep -Seconds 1
        $waited++
    }
    return -not (Test-ProcessAlive $ProcessId)
}

# ---------------------------------------------------------------- start

function Invoke-Start {
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    Clear-StalePid

    $p = Read-ServerPid
    if (Test-ProcessAlive $p) {
        Write-Log "이미 실행 중입니다 (PID $p, 포트 $Port)."
        return
    }

    if (-not (Get-Command $Mvn -ErrorAction SilentlyContinue)) {
        Die "'$Mvn' 을 찾을 수 없습니다. maven 을 설치하거나 MVN 환경변수로 경로를 지정하세요."
    }

    if (Test-PortInUse) {
        Die "포트 $Port 를 이미 다른 프로세스가 쓰고 있습니다 (PID: $(Get-PortListenerList)). 이 스크립트가 띄운 서버가 아니므로 직접 정리해 주세요."
    }

    Write-Log "기동합니다 (포트 $Port) ..."
    Set-Content -Path $LogFile -Value $null

    # cmd.exe 로 감싸서 stdout/stderr 를 한 로그 파일로 합친다(bash 의 `>> log 2>&1` 대응).
    # mvn 이 애플리케이션을 자식 JVM 으로 fork 하므로, PID 하나가 아니라 트리 전체를
    # taskkill /T 로 정리해야 한다.
    $cmdLine = "`"$Mvn`" spring-boot:run `"-Dspring-boot.run.arguments=--server.port=$Port`" >> `"$LogFile`" 2>&1"
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmdLine `
        -WorkingDirectory $AppDir -WindowStyle Hidden -PassThru

    Set-Content -Path $PidFile -Value $proc.Id

    # "떴다" 의 근거는 로그 문자열이 아니라 루트(/) 실제 응답이다.
    $waited = 0
    while ($waited -lt $StartTimeout) {
        if (Test-HealthOk) {
            Write-Log "기동 완료 — http://localhost:$Port (PID $($proc.Id), ${waited}초)"
            Write-Log "로그: $LogFile"
            return
        }
        if (-not (Test-ProcessAlive $proc.Id)) {
            Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
            Show-LogTail 30
            Die "기동 중 프로세스가 종료됐습니다."
        }
        Start-Sleep -Seconds 1
        $waited++
    }

    Show-LogTail 30
    Write-Warn "프로세스는 살아있지만 ${StartTimeout}초 안에 $HealthUrl 이 응답하지 않았습니다."
    Die "기동 확인 실패. 정리하려면 '.\server.ps1 stop' 을 실행하세요."
}

# ---------------------------------------------------------------- stop

function Invoke-Stop {
    $p = Read-ServerPid
    $stoppedSomething = $false

    if (Test-ProcessAlive $p) {
        $stoppedSomething = $true
        Write-Log "종료합니다 (PID $p) ..."
        if (-not (Stop-ProcessTreeAndWait -ProcessId $p -Timeout $StopTimeout)) {
            Write-Warn "${StopTimeout}초 안에 안 끝나서 강제 종료합니다."
            if (-not (Stop-ProcessTreeAndWait -ProcessId $p -Force -Timeout 10)) {
                Write-Warn "프로세스 $p 가 여전히 남아있습니다."
            }
        }
    }
    Remove-Item -Force $PidFile -ErrorAction SilentlyContinue

    # 안전망 — PID 기록이 어긋났거나 이전 실행이 남긴 프로세스가 포트를 물고 있을 수 있다.
    if (Test-PortInUse) {
        $pids = Get-PortListenerPids
        Write-Warn "포트 $Port 가 아직 잡혀 있습니다 (PID: $($pids -join ' ')). 잔여 프로세스를 정리합니다."
        foreach ($listenerPid in $pids) {
            & taskkill /PID $listenerPid /T *> $null
        }
        $waited = 0
        while ($waited -lt 10 -and (Test-PortInUse)) { Start-Sleep -Seconds 1; $waited++ }
        if (Test-PortInUse) {
            foreach ($listenerPid in (Get-PortListenerPids)) {
                & taskkill /PID $listenerPid /T /F *> $null
            }
            Start-Sleep -Seconds 1
        }
        if (Test-PortInUse) { Die "포트 $Port 를 비우지 못했습니다 (PID: $(Get-PortListenerList))." }
        $stoppedSomething = $true
    }

    if ($stoppedSomething) {
        Write-Log "종료 완료."
    } else {
        Write-Log "실행 중인 서버가 없습니다."
    }
}

# ---------------------------------------------------------------- restart / status

function Invoke-Restart {
    Invoke-Stop
    Invoke-Start
}

function Invoke-Status {
    $p = Read-ServerPid
    $listeners = Get-PortListenerList

    if (Test-ProcessAlive $p) {
        Write-Log "프로세스  실행 중 (PID $p)"
    } elseif ($p) {
        Write-Log "프로세스  없음 (PID 파일에 $p 가 남아있으나 죽은 상태)"
    } else {
        Write-Log "프로세스  없음"
    }

    if ($listeners) {
        Write-Log "포트      $Port LISTEN (PID: $listeners)"
    } else {
        Write-Log "포트      $Port 사용 안 함"
    }

    if (Test-HealthOk) {
        Write-Log "health    정상 ($HealthUrl)"
    } else {
        Write-Log "health    응답 없음 ($HealthUrl)"
    }

    Write-Log "로그      $LogFile"
}

# ---------------------------------------------------------------- 진입점

function Show-Usage {
    @"
사용법: server.ps1 {start|stop|restart|status}

  start     개발서버를 백그라운드로 띄우고 루트(/) 응답까지 기다린다
  stop      프로세스 트리 전체를 종료하고 포트를 비운다
  restart   stop 후 start
  status    프로세스 / 포트 / health 를 각각 확인한다

환경변수: PORT(기본 $Port) START_TIMEOUT($StartTimeout) STOP_TIMEOUT($StopTimeout) MVN($Mvn)
"@ | Write-Host
}

switch ($Command) {
    'start'   { Invoke-Start }
    'stop'    { Invoke-Stop }
    'restart' { Invoke-Restart }
    'status'  { Invoke-Status }
    default   { Show-Usage }
}
