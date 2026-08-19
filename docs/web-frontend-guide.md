# apps/web 프런트엔드 가이드

> 빌드 도구 없는 서버 사이드 렌더링(Jinja2) + 바닐라 JS 구조. 백엔드 API 계약은 [web-api-reference.md](./web-api-reference.md) 참고.

## 1. 스택 요약

- **템플릿 엔진**: Jinja2 (`frontend/router.py`가 `Jinja2Templates`로 렌더링)
- **CSS**: Tailwind — 빌드 파이프라인 없이 `static/vendor/tailwind.browser.js`(브라우저에서 직접 컴파일하는 Tailwind Play CDN 방식 번들)를 `<script>`로 로드. `base.html`의 `<style type="text/tailwindcss">` 블록에서 커스텀 테마 토큰을 선언한다.
- **폰트**: Pretendard Variable, CDN(`jsdelivr`)에서 로드.
- **JS**: 모듈 시스템 없는 바닐라 스크립트. 페이지마다 전용 JS 파일 하나를 `<script src>`로 붙인다. 번들러/트랜스파일 없음 — 브라우저가 그대로 실행.
- **아이콘**: 인라인 SVG (Lucide류 stroke 아이콘을 직접 마크업에 심음). 아이콘 라이브러리 의존성 없음.

## 2. 템플릿 구조

```
templates/
├── base.html        # 레이아웃 뼈대: 사이드바, 헤더(버전 배지, 테마 스위치), breadcrumb, {% block content %}
├── _nav.html         # 사이드바 메뉴 (6개 항목: Home/Predict/Training/Data Source/API Docs/Model Status)
├── index.html         # "/"  — 단일 이미지 업로드 예측 폼
├── predict.html        # "/predict" — 일괄 추론 (images/pred 대상 검증)
├── train.html           # "/train" — 학습 실행/모니터링
├── data_source.html      # "/data-source" — 캡차 이미지 수집 + 라벨링
└── status.html            # "/status" — 모델별 로드 상태 대시보드
```

모든 페이지는 `{% extends "base.html" %}`로 시작하고 `page_title`/`breadcrumb`/`content`/`scripts` 블록을 채운다. `_nav.html`의 활성 메뉴 표시는 각 라우터가 넘기는 `active_nav` 컨텍스트 값(`"predict"`, `"batch_predict"`, `"train"`, `"data_source"`, `"status"`)으로 결정된다.

페이지 컨텍스트는 `frontend/router.py`가 `services/*`를 호출해 채운다 — 템플릿 자체는 서버 상태를 조회하지 않는다. 예를 들어 `train.html`은 `services.train.list_targets()`가 만든 `targets` 리스트와 `PARAM_SPEC`(파라미터 이름·범위·기본값)을 그대로 폼 렌더링에 쓴다.

## 3. 디자인 토큰 & 다크 모드

`base.html`에 CSS 커스텀 프로퍼티로 TailAdmin 팔레트를 정의하고, Tailwind `@theme inline`에서 `--color-*` 변수로 매핑한다.

```css
:root { --background: #f9fafb; --surface: #ffffff; --brand: #465fff; ... }
.dark { --background: #101828; --surface: #101828; --brand: #465fff; ... }
```

색을 쓸 때는 원시 hex가 아니라 `bg-background`, `text-foreground`, `border-border`, `bg-brand`, `text-success` 같은 시맨틱 유틸리티 클래스를 쓴다. 새 색이 필요하면 `:root`와 `.dark` 양쪽에 변수를 추가하고 `@theme inline`에 매핑을 더한다.

다크 모드는 `<html>`의 `.dark` 클래스로 토글된다.

- **FOUC 방지**: `base.html` `<head>`의 인라인 스크립트가 페인트 전에 `localStorage.theme`을 읽어 클래스를 미리 적용한다.
- **런타임 전환**: `static/js/theme.js`가 헤더의 라이트/다크/시스템 3버튼을 처리하고 `localStorage`에 저장한다. `matchMedia("(prefers-color-scheme: dark)")` 변경도 구독해 "시스템" 선택 시 실시간 반영한다.
- 같은 파일이 반응형 사이드바 토글(`#sidebar-toggle`, `#overlay`)도 담당한다.

## 4. 정적 JS 파일과 담당 페이지

| 파일 | 페이지 | 역할 |
|---|---|---|
| `theme.js` | 전역 (`base.html`) | 다크모드, 사이드바 토글 |
| `app.js` | `index.html` (`/`) | 드래그앤드롭 업로드 → `POST /api/v1/predictImage` → 결과 표시 |
| `predict.js` | `predict.html` (`/predict`) | 일괄 추론 SSE 소비, 신뢰도 히스토그램, 오답 갤러리, 페이지네이션, 라이트박스 |
| `train.js` | `train.html` (`/train`) | 학습 시작/중단, 파라미터 폼 자동 저장, SSE 세션 재생, 손실 곡선(SVG) |
| `data_source.js` | `data_source.html` (`/data-source`) | 캡차/리비전 분리 셀렉트(리비전은 템플릿이 심은 `#data-source-targets` JSON으로 캡차별로 채우고 가장 높은 리비전을 기본 선택), 수집 SSE(`delay_ms` 포함), draft 갤러리, 인라인 라벨링 입력 |

각 파일은 IIFE나 모듈 래핑 없이 스크립트 최상위에서 `document.querySelector`로 DOM을 바로 참조한다 — 해당 템플릿에만 로드되므로 전역 스코프 충돌을 걱정할 필요가 없다(다른 페이지 JS와 함께 로드되지 않음).

## 5. SSE 클라이언트 패턴

세 페이지(`predict.js`, `train.js`, `data_source.js`)가 동일한 `EventSource` 패턴을 쓴다.

```js
const source = new EventSource(`/api/v1/<feature>/stream?${params}`);
source.addEventListener("start", (e) => { /* JSON.parse(e.data) */ });
source.addEventListener("item" /* or epoch/shuffle */, (e) => { /* ... */ });
source.addEventListener("summary" /* or done/skipped */, (e) => { /* 종료 처리, source.close() */ });
source.addEventListener("error", (event) => {
  // 서버가 error 이벤트를 보냈으면 event.data 가 있다.
  // 연결 자체가 끊겼으면 event.data 가 없고 readyState === CLOSED 다.
  if (event.data) { /* 서버 오류 메시지 표시 */ }
  else if (source.readyState === EventSource.CLOSED) { /* 연결 끊김 표시 */ }
});
```

핵심 관례:

- **종료 시 반드시 `source.close()`** — 세 파일 모두 `finish()` 헬퍼에서 처리한다. 브라우저의 `EventSource`는 서버 스트림이 끝나도 자동 재연결을 시도하므로, 명시적으로 닫지 않으면 계속 재접속을 시도한다.
- **중단 버튼은 스트림을 그냥 닫는 경우도 있고(`predict.js`, `data_source.js`), 별도 API로 정지 요청을 보내는 경우도 있다(`train.js`의 `POST /train/stop`)** — 학습은 에폭 경계까지 마저 돌아야 하므로 "중단 신호"와 "연결 종료"가 분리돼 있다.
- **재접속 시 히스토리 재생**: `train.js`의 `attachStream()`은 페이지 진입 시 `GET /train/targets`로 실행 중 여부를 먼저 확인하고, 실행 중이면 스트림에 붙어 서버가 재생하는 `start`~최신 이벤트를 그대로 받는다 (`services/train.py`의 세션 버퍼가 이를 지원).
- **렌더링 스로틀링**: `predict.js`는 `item` 이벤트마다 DOM을 다시 그리면 브라우저가 못 따라가므로 `requestAnimationFrame`으로 리페인트를 큐잉한다 (`scheduleRepaint()`).
- **DOM 상한**: `predict.js`의 오답 갤러리는 `GALLERY_LIMIT = 60`, `data_source.js`의 진행 표는 `ROW_LIMIT = 300`으로 무한정 쌓이는 것을 막는다. 갤러리류는 상한이 없는 경우도 있다(`data_source.js`의 라벨링 갤러리는 라벨을 붙이는 자리라 draft 전체를 보여줘야 하므로 상한 없음).

## 6. 서버 상태 → 폼 동기화 패턴 (`train.js`, `data_source.js`)

Training 페이지는 세 가지 폼 동기화 규칙을 쓴다. Data Source 페이지도 1·2번을 그대로 따른다(`GET/POST /api/v1/data-source/params`, `applyParams()`/`saveCurrentParams()`, 캡차·리비전 전환 시 `loadParamsForTarget()`). 유사한 상태-저장형 폼을 새로 만들 때 참고할 것.

1. **입력 변경 시 즉시 저장** — 각 필드에 `change` 리스너를 달아 `POST /api/v1/train/params`로 자동 저장한다. 저장 버튼이 따로 없다.
2. **프로그램적 값 설정은 저장을 재유발하지 않는다** — `applyParams()`가 `.value`/`.checked`를 직접 대입하면 `change` 이벤트가 발생하지 않으므로, 서버에서 불러온 값으로 폼을 채울 때 그 값이 다시 저장되는 재귀를 막는다.
3. **"실행 중" 상태와 "저장된 값" 상태를 섞지 않는다** — 페이지 진입 시 학습이 이미 돌고 있으면 스트림의 `start` 이벤트가 주는 "지금 실행 중인 파라미터"로 폼을 채우고, 아니면 `GET /train/params`가 주는 "마지막 저장값"으로 채운다. 두 경로를 동시에 타면 저장값이 실행 중 값을 덮어써 `epochs` 등이 실제와 어긋난다.

## 7. 새 페이지 추가하는 법

1. `templates/<name>.html` 생성, `{% extends "base.html" %}`로 `page_title`/`content`/`scripts` 블록 채움.
2. `frontend/router.py`에 라우트 추가 — 필요한 `services/*` 조회를 모아 템플릿 컨텍스트로 넘긴다 (`active_nav` 값 포함).
3. `templates/_nav.html`에 메뉴 항목 추가, `active_nav` 값과 매칭.
4. 동적 동작이 필요하면 `static/js/<name>.js`를 만들고 템플릿 하단 `{% block scripts %}`에서 로드.
5. 서버와 통신이 필요하면 `/api/v1` 아래 엔드포인트를 먼저 만든다 ([web-dev-guide.md](./web-dev-guide.md) §4 참고). 진행률이 있는 장시간 작업이면 §5의 SSE 패턴을 그대로 재사용한다.
6. 새 JS 파일에서 `fetch`/`EventSource` URL을 만들 때는 절대 경로 앞에 `CONTEXT_PATH`를 붙인다 (§9) — 안 붙이면 `WEB_CONTEXT_PATH` 설정 하위 경로 배포에서 요청이 프록시 밖 경로로 나가 깨진다.

## 8. 접근성 메모

- 아이콘 전용 버튼에는 `aria-label`을 붙인다 (`base.html`의 사이드바 토글, 테마 버튼 예시 참고).
- 사이드바/오버레이는 `role="group"`, `aria-pressed` 등 최소한의 ARIA 상태를 관리한다 (`theme.js`).
- 차트류(SVG 손실 곡선, 히스토그램)는 `role="img"` + `aria-label`로 스크린리더에 요약을 제공한다.
- 사용자 제공 문자열(파일명, 예측값)을 `innerHTML`에 끼울 때는 반드시 이스케이프한다 — `predict.js`의 `escapeHtml()` 참고. 새 코드에서 문자열을 마크업에 삽입할 때 이 패턴을 재사용할 것.

## 9. 리버스 프록시 하위 경로 (`CONTEXT_PATH`)

`WEB_CONTEXT_PATH`(예: `/captcha`)로 프록시 하위 경로에 배포하는 경우, 템플릿과 JS 양쪽에서 절대 경로 앞에 접두사를 직접 붙여야 한다. FastAPI의 `root_path`가 라우팅은 알아서 처리해도, HTML/JS에 박힌 `/static/...` 같은 절대 경로 문자열까지 자동으로 고쳐주지는 않기 때문이다 (백엔드 쪽 동작은 [web-architecture.md §8](./web-architecture.md#8-리버스-프록시-하위-경로-web_context_path) 참고).

- **템플릿**: `app.py`가 `templates.env.globals["context_path"]`로 값을 전역 주입한다. 모든 링크/자산 경로를 `{{ context_path }}/static/...`, `{{ context_path }}/` 형태로 쓴다 — `base.html`, `_nav.html`, 각 페이지 템플릿의 `<script src>` 태그가 이 패턴이다. `frontend/router.py`가 서버에서 렌더링하는 URL(예: `index.html`에 넘기는 `predict_image_url`)도 `f"{get_settings().web_context_path}/api/v1/predictImage"`처럼 직접 접두사를 붙인다.
- **정적 JS**: 템플릿 렌더링 문맥이 없으므로, `base.html`이 내려주는 `<html data-context-path="{{ context_path }}">` 속성을 읽는다. `train.js`/`predict.js`/`data_source.js` 맨 위에 있는 관례:

  ```js
  // 프록시 하위 경로 접두사(.env WEB_CONTEXT_PATH). base.html 이 <html data-context-path> 로 내려준다.
  const CONTEXT_PATH = document.documentElement.dataset.contextPath || "";
  ```

  이후 모든 `fetch`/`EventSource`/이미지 `src` URL 앞에 `` `${CONTEXT_PATH}/api/v1/...` ``처럼 붙인다.
- **예외**: `app.js`(index.html의 단일 예측 폼)는 별도 처리가 없다 — 폼의 `action` 속성 자체가 서버 렌더링 시점에 이미 접두사 붙은 값(`predict_image_url`)으로 채워지므로, JS는 `form.action`을 그대로 쓰기만 하면 된다.
- 새 페이지/JS를 추가할 때는 이 두 패턴(템플릿은 `context_path` 전역, JS는 `CONTEXT_PATH` 상수) 중 하나를 반드시 따른다. 하드코딩된 `/static/...`이나 `/api/v1/...` 절대 경로를 그대로 쓰면 하위 경로 배포에서 깨진다.
