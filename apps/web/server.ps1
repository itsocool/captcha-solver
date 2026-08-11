#Requires -Version 7.0
<#
.SYNOPSIS
	apps/web 개발서버(FastAPI + Uvicorn)를 백그라운드로 관리한다.

.DESCRIPTION
	start / stop / restart / status / logs 를 지원한다.

	주의할 점 두 가지가 설계에 반영돼 있다.

	1) `fastapi dev` 는 리로더(부모)와 워커(자식)로 뜬다. 부모만 종료하면 자식이
	   포트를 계속 물고 있어서 다음 start 가 실패한다. Stop 은 자식부터 훑어서
	   정리하고, 그래도 남으면 포트 점유자를 직접 찾아 끊는다.

	2) uvicorn 로그는 stdout 이 아니라 stderr 로 나간다. Start-Process 는 두
	   스트림을 같은 파일로 못 보내므로, pwsh 래퍼 안에서 `*>&1` 로 합쳐 한 파일에
	   적는다.

.EXAMPLE
	.\apps\web\server.ps1 start
	.\apps\web\server.ps1 status
	.\apps\web\server.ps1 logs -Follow
	.\apps\web\server.ps1 restart -Port 5001
#>

[CmdletBinding()]
param(
	[Parameter(Position = 0)]
	[ValidateSet('start', 'stop', 'restart', 'status', 'logs')]
	[string]$Command = 'status',

	# 생략하면 .env 의 WEB_PORT / WEB_HOST 를 쓴다.
	[int]$Port,
	[string]$BindHost,

	# start/restart 전용: 코드 변경 자동 반영을 끈다.
	[switch]$NoReload,

	# logs 전용
	[int]$Lines = 50,
	[switch]$Follow,

	# stop 전용: 포트를 물고 있는 프로세스가 이 스크립트가 띄운 게 아니어도 정리한다.
	[switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# apps/web/server.ps1 → 저장소 루트
$WebDir = $PSScriptRoot
$BaseDir = Split-Path (Split-Path $WebDir -Parent) -Parent

# server.sh 와 같은 위치·같은 파일명을 쓴다. 한 저장소에서 두 스크립트가 같은
# 서버를 다루는데 상태 파일이 흩어지면 헷갈린다 (.gitignore 도 한 줄로 끝난다).
$RunDir = Join-Path $WebDir '.dev'
$PidFile = Join-Path $RunDir 'server.pid'
$LogFile = Join-Path $RunDir 'server.log'

# server.sh 와 동일하게 uvicorn 을 직접 띄운다. `fastapi dev` 는 저장소 전체를
# 감시해서 captcha_data 수만 장에 걸려 리로드가 사실상 동작하지 않는다.
# 앱 임포트 경로는 web.app:app 이고, 작업 디렉터리는 저장소 루트여야 한다
# (core/config.py 의 db_path 등이 './db/...' 상대경로다).
$AppModule = 'web.app:app'
$ReloadDir = 'apps/web'

$STOP_TIMEOUT_SEC = 15
$START_TIMEOUT_SEC = 120


# ---------------------------------------------------------------- helpers

function Write-Step([string]$Message) { Write-Host "  $Message" -ForegroundColor DarkGray }
function Write-Ok([string]$Message) { Write-Host "  $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "  $Message" -ForegroundColor Yellow }
function Write-Err([string]$Message) { Write-Host "  $Message" -ForegroundColor Red }

function Read-EnvValue([string]$Name, [string]$Default) {
	$envPath = Join-Path $BaseDir '.env'
	if (Test-Path $envPath) {
		foreach ($line in Get-Content $envPath -Encoding utf8) {
			if ($line -match "^\s*$Name\s*=\s*(.*?)\s*$") {
				$value = $Matches[1].Trim('"').Trim("'")
				if ($value) { return $value }
			}
		}
	}
	return $Default
}

function Resolve-Settings {
	if (-not $script:Port) { $script:Port = [int](Read-EnvValue 'WEB_PORT' '5000') }
	if (-not $script:BindHost) { $script:BindHost = Read-EnvValue 'WEB_HOST' '0.0.0.0' }
}

function Get-DescendantPids([int]$ParentId) {
	# 재귀로 손자까지 모은다. fastapi dev 는 보통 pwsh → uv → python → python 이다.
	$found = @()
	$children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue)
	foreach ($child in $children) {
		$found += [int]$child.ProcessId
		$found += Get-DescendantPids ([int]$child.ProcessId)
	}
	return $found
}

function Get-PortOwnerPids([int]$TargetPort) {
	$conns = @(Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue)
	return @($conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { [int]$_ })
}

function Test-ProcessAlive([int]$ProcessId) {
	return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Get-SavedPid {
	if (-not (Test-Path $PidFile)) { return 0 }
	$raw = (Get-Content $PidFile -Raw -ErrorAction SilentlyContinue)
	if (-not $raw) { return 0 }
	$parsed = 0
	if ([int]::TryParse($raw.Trim(), [ref]$parsed)) { return $parsed }
	return 0
}

function Get-ServerState {
	<#
		실행 중 판단은 포트를 근거로 한다. PID 파일은 우리가 띄운 것인지 구분하는 보조
		수단일 뿐이다. 세션 중 부모가 죽고 자식만 남아 포트를 쥔 경우가 실제로 있었다.
	#>
	# @() 로 감싸는 게 중요하다. PowerShell 은 반환 시 원소 1개짜리 배열을 스칼라로
	# 풀어버려서, StrictMode 아래에서 .Count 접근이 터진다.
	$ownerPids = @(Get-PortOwnerPids $script:Port)
	$savedPid = Get-SavedPid

	return [pscustomobject]@{
		Running    = $ownerPids.Count -gt 0
		OwnerPids  = $ownerPids
		SavedPid   = $savedPid
		SavedAlive = ($savedPid -gt 0) -and (Test-ProcessAlive $savedPid)
		Port       = $script:Port
	}
}

function Get-Health([int]$TargetPort) {
	try {
		return Invoke-RestMethod -Uri "http://127.0.0.1:$TargetPort/health" -TimeoutSec 5 -ErrorAction Stop
	}
	catch {
		return $null
	}
}

function Stop-PidTree([int[]]$RootPids) {
	# 자식부터 죽인다. 부모를 먼저 죽이면 자식이 고아가 되어 포트를 계속 쥔다.
	$targets = @()
	foreach ($rootPid in $RootPids) {
		if ($rootPid -le 0) { continue }
		$targets += Get-DescendantPids $rootPid
		$targets += $rootPid
	}

	foreach ($target in ($targets | Select-Object -Unique)) {
		if (-not (Test-ProcessAlive $target)) { continue }
		try {
			Stop-Process -Id $target -Force -ErrorAction Stop
		}
		catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
			# 부모를 끊는 순간 자식이 같이 사라지는 경합. 목적은 달성됐으므로 조용히 넘어간다.
		}
		catch {
			Write-Warn "PID $target 종료 실패: $($_.Exception.Message)"
		}
	}
}

function Wait-PortReleased([int]$TargetPort, [int]$TimeoutSec) {
	$deadline = (Get-Date).AddSeconds($TimeoutSec)
	while ((Get-Date) -lt $deadline) {
		if (@(Get-PortOwnerPids $TargetPort).Count -eq 0) { return $true }
		Start-Sleep -Milliseconds 300
	}
	return $false
}


# ---------------------------------------------------------------- commands

function Invoke-Start {
	$state = Get-ServerState
	if ($state.Running) {
		Write-Warn "이미 실행 중입니다 (포트 $($state.Port), PID $($state.OwnerPids -join ', '))"
		Write-Step "재시작하려면: .\apps\web\server.ps1 restart"
		return 1
	}

	if (-not (Test-Path $RunDir)) { New-Item -ItemType Directory -Path $RunDir -Force | Out-Null }

	$banner = @(
		''
		('=' * 70)
		"[server.ps1] start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  host=$($script:BindHost) port=$($script:Port)"
		('=' * 70)
	) -join [Environment]::NewLine
	Add-Content -Path $LogFile -Value $banner -Encoding utf8

	$reloadArgs = if ($NoReload) { '' } else { "--reload --reload-dir $ReloadDir " }

	# uvicorn 은 stderr 로, 앱의 print() 는 stdout 으로 나간다. 2>&1 로 합쳐 한 파일에 적는다.
	#
	# Add-Content 를 파이프 끝에 두면 안 된다. 파이프라인이 끝날 때까지(= 서버가 죽을
	# 때까지) 기록이 안 나가서 로그가 계속 비어 있다. ForEach-Object 안에서 한 줄씩
	# 호출해야 즉시 flush 된다.
	# PYTHONUNBUFFERED 는 파이썬 쪽 블록 버퍼링을 끄는 용도다.
	$inner = "`$env:PYTHONUNBUFFERED = '1'; " +
	         "& uv run uvicorn $AppModule --host $($script:BindHost) --port $($script:Port) $($reloadArgs)2>&1 | " +
	         "ForEach-Object { Add-Content -LiteralPath '$LogFile' -Value `$_.ToString() -Encoding utf8 }"

	Write-Step "기동 중... (host=$($script:BindHost) port=$($script:Port) reload=$(if ($NoReload) { 'off' } else { 'on' }))"

	$proc = Start-Process -FilePath 'pwsh' `
		-ArgumentList '-NoProfile', '-NonInteractive', '-Command', $inner `
		-WorkingDirectory $BaseDir `
		-WindowStyle Hidden `
		-PassThru

	Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii

	# 모델 프리로드 때문에 기동에 시간이 걸린다. /health 가 응답할 때까지 기다린다.
	$deadline = (Get-Date).AddSeconds($START_TIMEOUT_SEC)
	while ((Get-Date) -lt $deadline) {
		if (-not (Test-ProcessAlive $proc.Id)) {
			Write-Err "기동 프로세스가 죽었습니다. 로그를 확인하세요:"
			Write-Step ".\apps\web\server.ps1 logs"
			return 1
		}
		$health = Get-Health $script:Port
		if ($health) {
			Write-Ok "기동 완료 (PID $($proc.Id), 포트 $($script:Port))"
			Write-Step "상태: $($health.status)  버전: $($health.version)  로드됨: $($health.loaded_captcha_ids -join ', ')"
			Write-Step "로그: $LogFile"
			return 0
		}
		Start-Sleep -Milliseconds 500
	}

	Write-Err "$START_TIMEOUT_SEC 초 안에 /health 응답이 없습니다"
	Write-Step ".\apps\web\server.ps1 logs 로 확인하세요"
	return 1
}

function Invoke-Stop {
	$state = Get-ServerState

	if (-not $state.Running -and -not $state.SavedAlive) {
		Write-Step "실행 중이 아닙니다 (포트 $($state.Port))"
		if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
		return 0
	}

	# PID 파일의 프로세스와 포트 점유자를 모두 대상으로 한다. 둘이 다를 수 있다.
	$roots = @()
	if ($state.SavedAlive) { $roots += $state.SavedPid }
	$roots += $state.OwnerPids
	$roots = @($roots | Where-Object { $_ -gt 0 } | Select-Object -Unique)

	Write-Step "종료 중... (PID $($roots -join ', '))"
	Stop-PidTree $roots

	if (-not (Wait-PortReleased $script:Port $STOP_TIMEOUT_SEC)) {
		# 부모가 죽고 고아 워커만 남은 경우. 포트 점유자를 다시 찾아 끊는다.
		$leftover = @(Get-PortOwnerPids $script:Port)
		if ($leftover.Count -gt 0) {
			Write-Warn "포트를 계속 점유하는 프로세스가 있습니다: $($leftover -join ', ')"
			Stop-PidTree $leftover
		}
		if (-not (Wait-PortReleased $script:Port 5)) {
			Write-Err "포트 $($script:Port) 를 해제하지 못했습니다"
			return 1
		}
	}

	if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
	Write-Ok "종료 완료 (포트 $($script:Port) 해제)"
	return 0
}

function Invoke-Status {
	$state = Get-ServerState

	Write-Host ""
	Write-Host "  apps/web 개발서버" -ForegroundColor Cyan
	Write-Host ('  ' + ('-' * 52)) -ForegroundColor DarkGray

	if (-not $state.Running) {
		Write-Host "  상태      : " -NoNewline; Write-Host "중지됨" -ForegroundColor Red
		Write-Host "  포트      : $($state.Port)"
		if ($state.SavedPid -gt 0 -and -not $state.SavedAlive) {
			Write-Warn "PID 파일에 죽은 PID $($state.SavedPid) 가 남아 있습니다"
		}
	}
	else {
		Write-Host "  상태      : " -NoNewline; Write-Host "실행 중" -ForegroundColor Green
		Write-Host "  포트      : $($state.Port)"
		Write-Host "  PID       : $($state.OwnerPids -join ', ')" -NoNewline
		if ($state.SavedPid -gt 0) { Write-Host "  (기록: $($state.SavedPid))" } else { Write-Host "  (PID 파일 없음 — 외부에서 띄운 서버)" }

		foreach ($ownerPid in $state.OwnerPids) {
			$proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
			if ($proc) {
				$uptime = (Get-Date) - $proc.StartTime
				Write-Host ("  가동 시간 : {0:hh\:mm\:ss}" -f $uptime)
				Write-Host ("  메모리    : {0:N0} MB" -f ($proc.WorkingSet64 / 1MB))
			}
		}

		$health = Get-Health $state.Port
		if ($health) {
			Write-Host "  health    : " -NoNewline
			Write-Host $health.status -ForegroundColor $(if ($health.status -eq 'ok') { 'Green' } else { 'Yellow' })
			Write-Host "  버전      : $($health.version)"
			Write-Host "  로드된 모델: $($health.loaded_captcha_ids -join ', ')"
		}
		else {
			Write-Warn "health 응답 없음 (기동 중이거나 오류 상태)"
		}
	}

	if (Test-Path $LogFile) {
		$log = Get-Item $LogFile
		Write-Host ("  로그      : {0} ({1:N0} KB, 최종 {2:HH:mm:ss})" -f $log.FullName, ($log.Length / 1KB), $log.LastWriteTime)
	}
	else {
		Write-Host "  로그      : (없음)"
	}
	Write-Host ""

	return $(if ($state.Running) { 0 } else { 1 })
}

function Invoke-Logs {
	if (-not (Test-Path $LogFile)) {
		Write-Warn "로그 파일이 없습니다: $LogFile"
		return 1
	}

	# Out-Host 가 중요하다. 그냥 두면 Get-Content 출력이 함수 반환값으로 흡수돼
	# 아래 switch 의 $exitCode 에 담기고, 화면에는 아무것도 안 나온다.
	if ($Follow) {
		Write-Step "$LogFile (Ctrl+C 로 종료)"
		Get-Content $LogFile -Tail $Lines -Wait -Encoding utf8 | Out-Host
	}
	else {
		Get-Content $LogFile -Tail $Lines -Encoding utf8 | Out-Host
	}
	return 0
}


# ---------------------------------------------------------------- main

Resolve-Settings

$exitCode = switch ($Command) {
	'start' { Invoke-Start }
	'stop' { Invoke-Stop }
	'restart' {
		$stopResult = Invoke-Stop
		if ($stopResult -ne 0) { $stopResult } else { Invoke-Start }
	}
	'status' { Invoke-Status }
	'logs' { Invoke-Logs }
}

exit $exitCode
