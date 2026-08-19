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

MAX_COUNT = 5000
REQUEST_TIMEOUT = 15.0
# 응답을 통째로 메모리에 올리므로 상한이 필요하다. 캡차 한 장은 보통 수 KB 라
# 8MB 면 충분히 넉넉하다.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

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
	레지스트리에 있는 캡차는 디렉터리가 아직 없어도 rev 0 으로 넣는다.
	"""
	from hypercaptcha import engine

	registered = engine.get_captcha_type_list()
	targets: list[dict] = []

	for captcha_id in sorted(registered):
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


def clean_request(captcha_id: str, rev: int, url: str, selector: str, count: int) -> dict:
	"""요청 값을 검증해 확정한다. 잘못된 값은 실행 오류가 아니라 요청 오류다."""
	from hypercaptcha import engine

	if captcha_id not in engine.get_captcha_type_list():
		raise ValueError(f"등록되지 않은 캡차입니다: {captcha_id!r}")

	if rev < 1:
		raise ValueError(f"rev 는 1 이상이어야 합니다 (받은 값 {rev})")

	parsed = urlparse((url or "").strip())
	if parsed.scheme not in ("http", "https") or not parsed.netloc:
		raise ValueError("URL 은 http:// 또는 https:// 로 시작하는 주소여야 합니다")

	try:
		count = int(count)
	except (TypeError, ValueError):
		raise ValueError(f"파일 개수가 숫자가 아닙니다 ({count!r})")
	if not (1 <= count <= MAX_COUNT):
		raise ValueError(f"파일 개수는 1 ~ {MAX_COUNT} 범위여야 합니다 (받은 값 {count})")

	return {
		"captcha_id": captcha_id,
		"rev": rev,
		"url": parsed.geturl(),
		"selector": (selector or "").strip(),
		"count": count,
	}


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


def _next_index(directory: Path) -> int:
	"""이어붙일 파일 번호. 기존에서 가장 큰 순번 + 1 이라 이름이 겹치지 않는다.

	숫자가 아닌 이름(라벨을 붙여 바꾼 파일)은 순번으로 세지 않는다.
	"""
	largest = 0
	for p in directory.glob("*.png"):
		if p.stem.isdigit():
			largest = max(largest, int(p.stem))
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
		buf = io.BytesIO()
		im.convert("RGB").save(buf, "PNG")
		return buf.getvalue()


def run(captcha_id: str, rev: int, url: str, selector: str, count: int,
        cancel: threading.Event | None = None):
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

	request = clean_request(captcha_id, rev, url, selector, count)

	if not _RUN_LOCK.acquire(blocking=False):
		raise DataSourceBusy("이미 다른 수집이 실행 중입니다")

	try:
		target_dir = draft_dir(request["captcha_id"], request["rev"])
		target_dir.mkdir(parents=True, exist_ok=True)
		index = _next_index(target_dir)

		yield {
			"type": "start",
			"captcha_id": request["captcha_id"], "rev": request["rev"],
			"url": request["url"], "selector": request["selector"],
			"total": request["count"], "draft_dir": str(target_dir),
			"existing": len(list(target_dir.glob("*.png"))), "start_index": index,
		}

		started = time.time()
		saved = failed = 0

		with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
			for i in range(request["count"]):
				if cancel is not None and cancel.is_set():
					break

				try:
					body, content_type = _fetch(client, request["url"])

					# URL 이 이미지 자체를 주면 셀렉터는 필요 없다.
					if content_type.startswith("image/"):
						content = body
						image_url = request["url"]
					else:
						# 셀렉터가 비면 페이지의 첫 이미지를 쓴다. 캡차 주소는 대개 그림
						# 한 장만 있는 페이지라(ipTIME 의 captcha.cgi) 이걸로 충분하다.
						html = body.decode("utf-8", errors="replace")
						image_url = _pick_image_url(html, request["url"], request["selector"] or "img")
						content, _ = _fetch(client, image_url)

					png = _decode_png(content)
				except Exception as e:
					failed += 1
					yield {
						"type": "item", "index": i, "saved": False,
						"error": f"{type(e).__name__}: {e}",
					}
					continue

				name = f"{index:06d}.png"
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
