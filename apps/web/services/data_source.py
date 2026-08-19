"""데이터 소스 수집(Data Source 페이지) 서비스.

services/batch_predict.py 와 같은 규칙이다. FastAPI 를 모르는 순수 파이썬이고,
평범한 dict 를 yield 하는 제너레이터라 서버 없이도 호출해서 확인할 수 있다.

캡차는 요청할 때마다 다른 그림을 주므로, 페이지를 count 번 다시 열어 매번 한 장씩
받는다. 받은 그림은 정답을 모르니 images/draft/ 에 쌓는다 (라벨 없는 원본을 두는
기존 관례). 라벨을 붙여 train/pred 로 옮기는 건 사람 몫이다.
"""

import re
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from web.core.config import BASE_DIR


# 네트워크와 디스크를 함께 쓴다. 같은 대상에 두 수집이 겹치면 파일명이 부딪히므로
# 서버 전체에서 한 번에 하나만 돌린다 (일괄 추론·학습과 같은 방식).
_RUN_LOCK = threading.Lock()

CAPTCHA_DATA_DIR = BASE_DIR / "captcha_data"

# 라벨이 아직 없는 draft 파일 이름. 수집 순번을 접두사와 함께 쓴다 — 저장소 관례가
# "파일 이름 = 정답" 이라 라벨을 붙이면 이름이 라벨로 바뀌는데, 캡차 대부분이 숫자
# 라벨이라 순번(000123)과 라벨(967238)이 구별되지 않았다. 접두사가 있어야 "아직 라벨
# 없음"을 판정할 수 있다 (예측으로 이름 변경 기능이 이미 라벨된 파일을 건드리지 않게).
DRAFT_NAME_RE = re.compile(r"^draft-(\d+)$")


def is_unlabeled_draft(name: str) -> bool:
	return DRAFT_NAME_RE.match(Path(name).stem) is not None


def draft_name(index: int) -> str:
	return f"draft-{index:06d}.png"

MAX_COUNT = 5000
# 요청 사이 지연 상한(ms). 대상 서버를 배려하려는 값이지 긴 대기용이 아니다.
# 지연 중에는 yield 가 없어 소비자가 끊어도 다음 yield 까지 락이 안 풀리므로 짧게 잡는다.
MAX_DELAY_MS = 30_000
REQUEST_TIMEOUT = 15.0

# 페이지 폼에서 (캡차, 리비전)별로 기억해 두는 입력값과 기본값. data_source_params 에
# JSON 으로 저장된다 (services/train.PERSIST_PARAMS 와 같은 방식). 템플릿의 초기값도 이와 같다.
# 응답 해석 방식. image: 본문이 이미지 그 자체. html: 페이지에서 CSS 셀렉터로 이미지 요소를
# 찾아 그 주소를 다시 받는다. json: 본문을 JSON 으로 읽고 셀렉터를 키 경로(예: data.image,
# items[0].url)로 써서 값을 꺼낸다 — 값은 이미지 URL, data: URI, base64 문자열 중 하나.
CONTENT_TYPES = ("image", "html", "json")
DEFAULT_PARAMS = {"url": "", "content_type": "image", "selector": "", "count": 500, "delay_ms": 0}
PERSIST_PARAMS = tuple(DEFAULT_PARAMS)
# 응답을 통째로 메모리에 올리므로 상한이 필요하다. 캡차 한 장은 보통 수 KB 라
# 8MB 면 충분히 넉넉하다.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
# 브라우저처럼 보이는 요청 헤더. 대법원(ssgo.scourt.go.kr) 등 공공기관 WAF 는 기본
# python-httpx/x.y User-Agent 를 "보안 정책 위반"으로 막고 HTML 안내 페이지를 준다.
REQUEST_HEADERS = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
	"Accept": "image/avif,image/webp,image/png,image/*,application/json,text/html,*/*;q=0.8",
	"Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

# 이 서비스는 운영자가 넣은 URL 을 서버가 대신 요청한다. 사설/루프백 대역을 막자는
# 통상적인 SSRF 대책은 여기서 쓸 수 없다 — 주 용도가 ipTIME 공유기(192.168.x.x)처럼
# LAN 안에 있는 기기에서 캡차를 받아오는 것이라, 막으면 기능이 성립하지 않는다.
# 대신 지켜야 할 선은 배포 쪽이다: 이 웹앱에는 인증이 없으므로 신뢰할 수 없는
# 네트워크에 열어두면 안 된다 (기본 web_host=0.0.0.0 이니 방화벽/리버스 프록시로 막을 것).
# 여기서는 메모리를 지키는 응답 크기 상한만 건다.

# 같은 그림이 와도 거르지 않는다. 캡차는 글자가 같아도 노이즈가 매번 달라 내용 해시가
# 사실상 걸러주지 못하는데, 수백 장을 매번 다시 읽는 비용만 든다. 겹치는 그림은
# 라벨을 붙이는 사람이 판단한다.


class DataSourceBusy(Exception):
	"""이미 다른 수집이 돌고 있다."""


class DataSourceError(Exception):
	pass


def is_running() -> bool:
	return _RUN_LOCK.locked()


def list_targets() -> list[dict]:
	"""수집 대상으로 고를 수 있는 (캡차, 리비전) 목록.

	학습과 달리 이미지가 없어도 고를 수 있어야 한다 — 없어서 모으는 것이다.
	레지스트리에 있는 캡차는 디렉터리가 아직 없어도 레지스트리 rev(1부터 시작)로 넣는다.
	"""
	from hypercaptcha import engine
	from web.services.captcha import ordered_captcha_ids

	registered = engine.get_captcha_type_list()
	targets: list[dict] = []

	for captcha_id in ordered_captcha_ids(registered):
		captcha_dir = CAPTCHA_DATA_DIR / captcha_id
		revs = sorted(
			int(d.name) for d in captcha_dir.iterdir()
			if captcha_dir.is_dir() and d.is_dir() and d.name.isdigit()
		) if captcha_dir.is_dir() else []
		if not revs:
			revs = [registered[captcha_id].train_data.rev]

		for rev in revs:
			draft = CAPTCHA_DATA_DIR / captcha_id / str(rev) / "images" / "draft"
			targets.append({
				"captcha_id": captcha_id,
				"name": registered[captcha_id].name,
				"rev": rev,
				"draft_count": len(list(draft.glob("*.png"))) if draft.is_dir() else 0,
			})

	return targets


def clean_request(captcha_id: str, rev: int, url: str, selector: str, count: int,
                  delay_ms: int = 0, content_type: str = "image") -> dict:
	"""요청 값을 검증해 확정한다. 잘못된 값은 실행 오류가 아니라 요청 오류다."""
	from hypercaptcha import engine

	if captcha_id not in engine.get_captcha_type_list():
		raise ValueError(f"등록되지 않은 캡차입니다: {captcha_id!r}")

	if rev < 1:
		raise ValueError(f"rev 는 1 이상이어야 합니다 (받은 값 {rev})")

	parsed = urlparse((url or "").strip())
	if parsed.scheme not in ("http", "https") or not parsed.netloc:
		raise ValueError("URL 은 http:// 또는 https:// 로 시작하는 주소여야 합니다")

	content_type = _clean_content_type(content_type)
	selector = (selector or "").strip()
	if content_type == "json" and not selector:
		raise ValueError("JSON 응답은 이미지 값의 키 경로가 필요합니다 (예: data.image)")

	try:
		count = int(count)
	except (TypeError, ValueError):
		raise ValueError(f"파일 개수가 숫자가 아닙니다 ({count!r})")
	if not (1 <= count <= MAX_COUNT):
		raise ValueError(f"파일 개수는 1 ~ {MAX_COUNT} 범위여야 합니다 (받은 값 {count})")

	try:
		delay_ms = int(delay_ms or 0)
	except (TypeError, ValueError):
		raise ValueError(f"지연(ms)이 숫자가 아닙니다 ({delay_ms!r})")
	if not (0 <= delay_ms <= MAX_DELAY_MS):
		raise ValueError(f"지연(ms)은 0 ~ {MAX_DELAY_MS} 범위여야 합니다 (받은 값 {delay_ms})")

	return {
		"captcha_id": captcha_id,
		"rev": rev,
		"url": parsed.geturl(),
		"content_type": content_type,
		"selector": selector,
		"count": count,
		"delay_ms": delay_ms,
	}


def _clean_content_type(value) -> str:
	value = (value or "image").strip().lower()
	if value not in CONTENT_TYPES:
		raise ValueError(f"content_type 은 {'/'.join(CONTENT_TYPES)} 중 하나여야 합니다 (받은 값 {value!r})")
	return value


def clean_params(raw: dict) -> dict:
	"""폼 저장용 검증. 실행(clean_request)과 달리 url 은 비어 있어도 된다 —
	아직 주소를 안 넣고 개수/지연만 고쳐도 저장돼야 하기 때문이다. 넣었다면 http(s) 여야 한다."""
	params = dict(DEFAULT_PARAMS)

	url = (raw.get("url") or "").strip()
	if url:
		parsed = urlparse(url)
		if parsed.scheme not in ("http", "https") or not parsed.netloc:
			raise ValueError("URL 은 http:// 또는 https:// 로 시작하는 주소여야 합니다")
		url = parsed.geturl()
	params["url"] = url
	params["content_type"] = _clean_content_type(raw.get("content_type"))
	params["selector"] = (raw.get("selector") or "").strip()

	count = raw.get("count")
	if count not in (None, ""):
		try:
			count = int(count)
		except (TypeError, ValueError):
			raise ValueError(f"파일 개수가 숫자가 아닙니다 ({count!r})")
		if not (1 <= count <= MAX_COUNT):
			raise ValueError(f"파일 개수는 1 ~ {MAX_COUNT} 범위여야 합니다 (받은 값 {count})")
		params["count"] = count

	delay_ms = raw.get("delay_ms")
	if delay_ms not in (None, ""):
		try:
			delay_ms = int(delay_ms)
		except (TypeError, ValueError):
			raise ValueError(f"지연(ms)이 숫자가 아닙니다 ({delay_ms!r})")
		if not (0 <= delay_ms <= MAX_DELAY_MS):
			raise ValueError(f"지연(ms)은 0 ~ {MAX_DELAY_MS} 범위여야 합니다 (받은 값 {delay_ms})")
		params["delay_ms"] = delay_ms

	return params


def load_params(captcha_id: str, rev: int) -> dict:
	"""대상의 저장된 수집 입력값. 없으면 기본값. 저장 스키마가 바뀌어도 아는 키만 받는다."""
	from web.core.db import get_data_source_params

	params = dict(DEFAULT_PARAMS)
	stored = get_data_source_params(captcha_id, rev) or {}
	for key in PERSIST_PARAMS:
		if key in stored:
			params[key] = stored[key]
	return params


def _persist_params(captcha_id: str, rev: int, params: dict) -> None:
	from web.core.db import save_data_source_params

	save_data_source_params(captcha_id, rev, {k: params[k] for k in PERSIST_PARAMS if k in params})


def save_params(captcha_id: str, rev: int, raw: dict) -> dict:
	"""폼 입력값을 검증·저장한다. 수집을 시작하지 않아도 대상별로 유지된다.

	실행 시점에만 저장하면 '값만 바꾸고 페이지를 떠나면' 사라진다. 편집 시 프런트가 호출한다.
	"""
	from hypercaptcha import engine

	if captcha_id not in engine.get_captcha_type_list():
		raise ValueError(f"등록되지 않은 캡차입니다: {captcha_id!r}")
	if rev < 1:
		raise ValueError(f"rev 는 1 이상이어야 합니다 (받은 값 {rev})")
	params = clean_params(raw)
	_persist_params(captcha_id, rev, params)
	return params


def draft_dir(captcha_id: str, rev: int) -> Path:
	return CAPTCHA_DATA_DIR / captcha_id / str(rev) / "images" / "draft"


def draft_image_path(captcha_id: str, rev: int, image_name: str) -> Path:
	"""썸네일로 내보낼 draft 이미지의 실제 경로.

	batch_predict.pred_image_path 와 같은 규칙이다 — image_name 은 클라이언트가 준
	값이라 그대로 믿으면 경로 탈출이 된다. basename 만 취하고 확장자를 png 로
	제한한 뒤, 최종 경로가 draft 디렉터리 안인지 한 번 더 확인한다.
	"""
	import os

	safe_name = os.path.basename(image_name)
	if not safe_name or not safe_name.lower().endswith(".png"):
		raise ValueError("png 파일명만 허용합니다")

	base = draft_dir(captcha_id, rev).resolve()
	path = (base / safe_name).resolve()

	if not path.is_relative_to(base):
		raise ValueError("허용되지 않은 경로입니다")
	if not path.is_file():
		raise ValueError("이미지를 찾을 수 없습니다")

	return path


def list_drafts(captcha_id: str, rev: int, limit: int | None = None) -> dict:
	"""draft 에 쌓인 이미지 목록. 만들어진 순서(오래된 것부터)다.

	이름순으로 정렬하면 라벨을 붙여 이름이 바뀔 때마다 자리가 튄다. 받은 순서는
	라벨을 붙여도 그대로라 보고 있던 자리가 유지된다 (rename 은 mtime 을 건드리지
	않는다). 같은 시각이면 이름으로 가른다.
	limit 을 주지 않으면 전부 준다 — 라벨을 붙이려면 다 보여야 한다.
	"""
	directory = draft_dir(captcha_id, rev)
	names = [
		p.name for p in sorted(directory.glob("*.png"), key=lambda p: (p.stat().st_mtime, p.name))
	] if directory.is_dir() else []
	return {
		"names": names[:limit] if limit else names,
		"total": len(names),
		"unlabeled": sum(1 for n in names if is_unlabeled_draft(n)),
		"draft_dir": str(directory),
	}


def rename_draft(captcha_id: str, rev: int, name: str, label: str) -> dict:
	"""draft 이미지에 라벨을 붙인다.

	저장소 관례가 '파일 이름 = 정답' 이라(train/pred 를 보면 aaarv.png 식이다)
	이름을 바꾸는 것이 곧 라벨링이다. 경로 탈출 방어는 draft_image_path 가 한다.
	"""
	source = draft_image_path(captcha_id, rev, name)

	label = (label or "").strip()
	if not label:
		raise ValueError("라벨이 비었습니다")
	if any(c in label for c in "/\\\0") or label in (".", ".."):
		raise ValueError(f"파일 이름으로 쓸 수 없는 라벨입니다: {label!r}")

	target = source.with_name(f"{label}.png")
	if target == source:
		return {"name": source.name, "renamed": False}
	# rename 은 있는 파일을 조용히 덮어쓴다. 먼저 막는다.
	if target.exists():
		raise ValueError(f"이미 같은 이름이 있습니다: {target.name}")

	source.rename(target)
	return {"name": target.name, "renamed": True}


def iter_auto_label(captcha_id: str, rev: int, device: str | None = None,
                    min_confidence: float = 0.0):
	"""draft 이미지를 모델 예측값으로 이름 바꾸는(=라벨 붙이는) 이벤트 제너레이터.

	라벨이 아직 없는 파일(이름이 draft-순번인 것)만 대상이고, 이미 라벨이 붙은 파일은
	건너뛴다. 예측 신뢰도가 min_confidence 보다 낮으면 바꾸지 않고 low_confidence 로
	남긴다. 같은 이름이 이미 있으면 rename_draft 가 ValueError 를 내고 그 건은 실패로
	기록한다. 수집(run)과 같은 draft 디렉터리를 만지므로 같은 락을 쓴다.

	이벤트: start {total, device} → item {name, new_name, prediction, confidence, renamed,
	skipped?, error?} (매 장) → summary {total, renamed, skipped, failed}.
	"""
	from hypercaptcha import engine
	from web.services.captcha import get_model

	if captcha_id not in engine.get_captcha_type_list():
		raise ValueError(f"등록되지 않은 캡차입니다: {captcha_id!r}")
	if rev < 1:
		raise ValueError(f"rev 는 1 이상이어야 합니다 (받은 값 {rev})")
	try:
		min_confidence = float(min_confidence or 0.0)
	except (TypeError, ValueError):
		raise ValueError(f"신뢰도 하한이 숫자가 아닙니다 ({min_confidence!r})")
	if not (0.0 <= min_confidence <= 1.0):
		raise ValueError(f"신뢰도 하한은 0 ~ 1 범위여야 합니다 (받은 값 {min_confidence})")

	if not _RUN_LOCK.acquire(blocking=False):
		raise DataSourceBusy("이미 다른 수집/라벨링이 실행 중입니다")

	try:
		names = list_drafts(captcha_id, rev)["names"]
		targets = [n for n in names if is_unlabeled_draft(n)]
		model = get_model(captcha_id, device)  # 잘못된 디바이스는 여기서 ValueError
		yield {"type": "start", "captcha_id": captcha_id, "rev": rev,
		       "total": len(targets), "already_labeled": len(names) - len(targets),
		       "device": str(model.device), "min_confidence": min_confidence}

		renamed = skipped = failed = 0
		for i, name in enumerate(targets):
			try:
				path = draft_image_path(captcha_id, rev, name)
				prediction, confidence = engine.predict(model=model, image_path=str(path), verbose=0)
				confidence = float(confidence)
				if confidence < min_confidence:
					skipped += 1
					yield {"type": "item", "index": i, "name": name, "new_name": name,
					       "prediction": prediction, "confidence": confidence,
					       "renamed": False, "skipped": "low_confidence"}
					continue
				result = rename_draft(captcha_id, rev, name, prediction)
				renamed += 1 if result["renamed"] else 0
				yield {"type": "item", "index": i, "name": name, "new_name": result["name"],
				       "prediction": prediction, "confidence": confidence,
				       "renamed": result["renamed"]}
			except Exception as e:
				failed += 1
				yield {"type": "item", "index": i, "name": name, "new_name": name,
				       "renamed": False, "error": f"{type(e).__name__}: {e}"}

		yield {"type": "summary", "total": len(targets), "renamed": renamed,
		       "skipped": skipped, "failed": failed}
	finally:
		_RUN_LOCK.release()


def _next_index(directory: Path) -> int:
	"""이어붙일 파일 번호. 기존에서 가장 큰 순번 + 1 이라 이름이 겹치지 않는다.

	draft-NNNNNN 형식이 아닌 이름(라벨을 붙여 바꾼 파일)은 순번으로 세지 않는다.
	"""
	largest = 0
	for p in directory.glob("*.png"):
		match = DRAFT_NAME_RE.match(p.stem)
		if match:
			largest = max(largest, int(match.group(1)))
	return largest + 1


def _pick_image_url(html: str, page_url: str, selector: str) -> str:
	"""CSS 셀렉터로 고른 요소에서 이미지 주소를 뽑아 절대 경로로 만든다."""
	from bs4 import BeautifulSoup

	soup = BeautifulSoup(html, "html.parser")
	found = soup.select(selector)
	if not found:
		raise DataSourceError(f"셀렉터에 맞는 요소가 없습니다: {selector!r}")

	node = found[0]
	# img 는 src, 그 밖의 태그는 배경 이미지나 data 속성에 들어있는 경우가 많다.
	src = node.get("src") or node.get("data-src") or node.get("href")
	if not src:
		style = node.get("style") or ""
		match = re.search(r"url\(\s*['\"]?(.*?)['\"]?\s*\)", style)
		src = match.group(1) if match else None
	if not src:
		raise DataSourceError(
			f"요소 <{node.name}> 에서 이미지 주소를 찾지 못했습니다 (src/data-src/href/background 없음)"
		)

	return urljoin(page_url, src)


def _json_path(data, path: str):
	"""점 표기 키 경로로 JSON 값을 꺼낸다. 'a.b[0].c' 와 'a.b.0.c' 둘 다 받는다."""
	current = data
	for raw in path.replace("[", ".").replace("]", "").split("."):
		key = raw.strip()
		if key == "":
			continue
		if isinstance(current, list):
			if not key.lstrip("-").isdigit():
				raise DataSourceError(f"키 경로 {path!r}: 배열에는 숫자 인덱스가 와야 합니다 ({key!r})")
			try:
				current = current[int(key)]
			except IndexError:
				raise DataSourceError(f"키 경로 {path!r}: 배열 인덱스 {key} 가 범위를 벗어났습니다")
		elif isinstance(current, dict):
			if key not in current:
				raise DataSourceError(f"키 경로 {path!r}: {key!r} 키가 없습니다 (있는 키: {', '.join(map(str, current))[:200]})")
			current = current[key]
		else:
			raise DataSourceError(f"키 경로 {path!r}: {key!r} 앞에서 객체/배열이 아닌 값을 만났습니다")
	return current


def _pick_image_from_json(client, body: bytes, page_url: str, path: str) -> tuple[bytes, str]:
	"""JSON 응답에서 키 경로로 이미지 값을 꺼낸다.

	값은 세 가지 중 하나다: data: URI(그 자리에서 디코딩), 이미지 URL(다시 받음, 상대 경로면
	요청 URL 기준), 순수 base64 문자열(디코딩). (bytes, 표시용 출처) 를 돌려준다.
	"""
	import base64
	import json as _json

	try:
		data = _json.loads(body.decode("utf-8", errors="replace"))
	except ValueError as e:
		raise DataSourceError(f"JSON 으로 읽을 수 없습니다: {e}")

	value = _json_path(data, path)
	if not isinstance(value, str) or not value.strip():
		raise DataSourceError(f"키 경로 {path!r} 의 값이 문자열이 아닙니다 ({type(value).__name__})")
	value = value.strip()

	if value.startswith("data:"):
		header, _, payload = value.partition(",")
		if ";base64" not in header:
			raise DataSourceError("data: URI 가 base64 가 아닙니다")
		return base64.b64decode(payload, validate=False), f"{page_url}#{path}"

	if value.startswith(("http://", "https://", "/", "./", "../")):
		image_url = urljoin(page_url, value)
		content, _ = _fetch(client, image_url)
		return content, image_url

	try:
		# 순수 base64. 개행/공백이 섞여 있어도 받는다.
		return base64.b64decode("".join(value.split()), validate=True), f"{page_url}#{path}"
	except Exception:
		raise DataSourceError(f"키 경로 {path!r} 의 값이 이미지 URL/data URI/base64 가 아닙니다")


def _fetch(client, url: str) -> tuple[bytes, str]:
	"""응답을 크기 상한 안에서 받아 (본문, content-type) 으로 돌려준다.

	httpx 의 response.content 는 무조건 전부 읽으므로 스트리밍으로 받으면서 센다.
	"""
	with client.stream("GET", url) as response:
		response.raise_for_status()
		content_type = response.headers.get("content-type", "")

		declared = response.headers.get("content-length")
		if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
			raise DataSourceError(f"응답이 너무 큽니다 ({declared} bytes > {MAX_RESPONSE_BYTES})")

		chunks, total = [], 0
		for chunk in response.iter_bytes():
			total += len(chunk)
			if total > MAX_RESPONSE_BYTES:
				raise DataSourceError(f"응답이 {MAX_RESPONSE_BYTES} bytes 를 넘어 중단했습니다")
			chunks.append(chunk)

	return b"".join(chunks), content_type


def _decode_png(content: bytes) -> bytes:
	"""받은 바이트가 진짜 이미지인지 확인하고 PNG 로 통일해 돌려준다.

	저장소 파이프라인이 *.png 만 훑기 때문이다 (iptime 원본이 GIF 라 변환이 필요했다).
	"""
	import io

	from PIL import Image

	with Image.open(io.BytesIO(content)) as im:
		im.load()
		# 투명 배경(RGBA/LA/투명 팔레트)은 흰 배경에 합성한다. convert("RGB") 만 하면 투명
		# 픽셀의 RGB(보통 검정)가 그대로 남아 배경이 검게 저장된다 — 대법원 캡차가 그렇다.
		# 학습 데이터와 TrainData 전처리(RGBA → 흰 배경)와 같은 규약이다.
		if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
			rgba = im.convert("RGBA")
			background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
			im = Image.alpha_composite(background, rgba)
		buf = io.BytesIO()
		im.convert("RGB").save(buf, "PNG")
		return buf.getvalue()


def _sleep_ms(delay_ms: int, cancel: threading.Event | None = None) -> None:
	"""요청 사이 지연. cancel 이 오면 남은 시간을 기다리지 않는다."""
	if cancel is not None:
		cancel.wait(delay_ms / 1000.0)
	else:
		time.sleep(delay_ms / 1000.0)


def run(captcha_id: str, rev: int, url: str, selector: str, count: int,
        delay_ms: int = 0, content_type: str = "image", cancel: threading.Event | None = None):
	"""수집 이벤트 제너레이터.

	이벤트는 셋이다:
	  {'type': 'start',   ...}  맨 처음 한 번. total 로 받을 장수를 알려준다.
	  {'type': 'item',    ...}  매 장마다. 실패도 이 이벤트로 알린다.
	  {'type': 'summary', ...}  맨 마지막 한 번.

	학습과 달리 작업이 이 제너레이터 안에서 돌기 때문에, 소비자가 끊으면 다음 yield
	에서 GeneratorExit 가 도착해 finally 가 정상 실행된다 (services/train.py 의 주석
	참고 — 그쪽은 큐 대기로 블로킹돼 close 가 도달하지 못한다).
	"""
	import httpx

	request = clean_request(captcha_id, rev, url, selector, count, delay_ms, content_type)
	# 실행에 쓴 값은 그 대상의 마지막 입력값으로 남긴다.
	_persist_params(request["captcha_id"], request["rev"], request)

	if not _RUN_LOCK.acquire(blocking=False):
		raise DataSourceBusy("이미 다른 수집이 실행 중입니다")

	try:
		target_dir = draft_dir(request["captcha_id"], request["rev"])
		target_dir.mkdir(parents=True, exist_ok=True)
		index = _next_index(target_dir)

		yield {
			"type": "start",
			"captcha_id": request["captcha_id"], "rev": request["rev"],
			"url": request["url"], "content_type": request["content_type"], "selector": request["selector"],
			"total": request["count"], "delay_ms": request["delay_ms"],
			"draft_dir": str(target_dir),
			"existing": len(list(target_dir.glob("*.png"))), "start_index": index,
		}

		started = time.time()
		saved = failed = 0

		with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True, headers=REQUEST_HEADERS) as client:
			for i in range(request["count"]):
				if cancel is not None and cancel.is_set():
					break
				if i > 0 and request["delay_ms"] > 0:
					_sleep_ms(request["delay_ms"], cancel)
					if cancel is not None and cancel.is_set():
						break

				try:
					body, _ = _fetch(client, request["url"])

					if request["content_type"] == "image":
						# URL 이 이미지 자체를 준다. 셀렉터는 쓰지 않는다.
						content = body
						image_url = request["url"]
					elif request["content_type"] == "html":
						# 셀렉터가 비면 페이지의 첫 이미지를 쓴다. 캡차 주소는 대개 그림
						# 한 장만 있는 페이지라 이걸로 충분하다.
						html = body.decode("utf-8", errors="replace")
						image_url = _pick_image_url(html, request["url"], request["selector"] or "img")
						content, _ = _fetch(client, image_url)
					else:
						content, image_url = _pick_image_from_json(
							client, body, request["url"], request["selector"])

					png = _decode_png(content)
				except Exception as e:
					failed += 1
					yield {
						"type": "item", "index": i, "saved": False,
						"error": f"{type(e).__name__}: {e}",
					}
					continue

				name = draft_name(index)
				(target_dir / name).write_bytes(png)
				index += 1
				saved += 1
				yield {"type": "item", "index": i, "saved": True, "name": name,
				       "image_url": image_url, "bytes": len(png)}

		yield {
			"type": "summary",
			"requested": request["count"], "saved": saved,
			"failed": failed,
			"draft_dir": str(target_dir),
			"draft_total": len(list(target_dir.glob("*.png"))),
			"elapsed_sec": time.time() - started,
		}
	finally:
		_RUN_LOCK.release()
