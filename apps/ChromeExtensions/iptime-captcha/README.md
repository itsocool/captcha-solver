# ipTIME Captcha Solver (Chrome MV3 확장)

ipTIME 공유기 로그인 페이지의 캡차 이미지를 감지해 captcha-solver API 로
자동 인식하고, 결과를 캡차 입력칸에 채워 넣는 Chrome 확장(Manifest V3)이다.

- 대상 페이지: `http://192.168.50.4/sess-bin/login_session.cgi`
- 캡차: top document 의 `<img>` 가 아니라 **iframe** 안에 있다.
  - `#iframe_captcha` (src `/sess-bin/captcha.cgi`, 201x70)
  - 페이지와 same-origin → `contentDocument` 접근 및 canvas 읽기 가능

## 동작

1. content script 가 위 URL 에서 캡차 iframe(`#iframe_captcha`)과 그 내부
   이미지를 찾으면 background 에 보고하고, 해당 탭의 툴바 아이콘이 **활성화**되며
   캡차 테두리가 **녹색 점선**으로 바뀐다. (그 외 페이지에서는 비활성)
2. content script 가 iframe 내부 이미지를 canvas 로 그려 PNG data URL 로 변환한다.
3. background service worker 가 이를 Blob 으로 바꿔 solver API 에 POST 한다.
   - `POST {API_BASE}/api/v1/predictImage`
   - multipart: `captcha_id=iptime`, `image=<png>`
   - 응답: `{ captcha_id, prediction, confidence, elapsed_ms, device }`
4. 입력칸의 안내 문구(배경 이미지)를 `focus()` 로 걷어낸 뒤 인식 결과를 채운다.
   포커스는 캡차 칸에 남으므로 값 확인 후 바로 Enter 로 로그인할 수 있다.
5. 인식 결과를 클립보드에 복사하고, 화면 우상단
   플로팅 버튼에 표시한다. **로그인 폼을 자동 제출하지는 않는다.**
6. 플로팅 버튼(`🔓 캡차 자동 인식`)을 클릭하면 언제든 다시 인식할 수 있다.

## 설치 (Load unpacked)

1. Chrome 에서 `chrome://extensions` 열기
2. 우상단 **개발자 모드(Developer mode)** 켜기
3. **압축해제된 확장 프로그램을 로드합니다(Load unpacked)** 클릭
4. `apps/ChromeExtensions/iptime-captcha` 폴더 선택
5. 대상 로그인 페이지로 이동하면 툴바 아이콘이 활성화된다.
   (아이콘 파일이 없어 Chrome 기본 아이콘이 표시된다.)

## 설정 (config knobs)

| 상수 | 위치 | 설명 |
|------|------|------|
| `API_BASE` | `background.js` | solver API 베이스 주소. 기본 `http://192.168.50.98:30008`. HTTPS 프록시 사용 시 `https://dev.hyperinfo.co.kr:12005` 로 변경하고, `manifest.json` 의 `host_permissions` 에도 해당 도메인을 추가해야 한다. |
| `CAPTCHA_ID` | `background.js` | 캡차 유형 식별자. 기본 `iptime`. |
| `CAPTCHA_INPUT_SELECTOR` | `content.js` | 캡차 입력칸 CSS 셀렉터. 기본 `input[name="captcha_code"]`. 비우면 캡차 iframe 에서 조상을 타고 바깥으로 넓혀가며 빈 텍스트 입력칸을 찾는 휴리스틱으로 폴백한다. |
| `IFRAME_SELECTOR` | `content.js` | 캡차 iframe 셀렉터. 기본 `#iframe_captcha`. |

## 사용자가 반드시 확인해야 할 가정

- **캡차 입력칸 셀렉터**: ipTIME 펌웨어 버전마다 로그인 폼 구조가 다르다.
  자동 휴리스틱이 엉뚱한 칸을 채우면, 페이지에서 캡차 입력칸을 우클릭 →
  검사(Inspect)로 정확한 셀렉터를 확인해 `content.js` 의
  `CAPTCHA_INPUT_SELECTOR` 에 지정하라. (예: `input[name="captcha"]`)
- **API 베이스 도달성**: `API_BASE` 주소가 **브라우저가 접속한 네트워크에서**
  실제로 열려 있어야 한다. 공유기와 solver 서버가 서로 다른 망에 있으면
  HTTP 직접 주소 대신 HTTPS 프록시(`dev.hyperinfo.co.kr:12005`)를 쓰고
  `host_permissions` 를 맞춰라.
- **캡차 iframe 셀렉터**: `#iframe_captcha` 와 그 내부의 첫 `img` 가 실제 캡차라고
  가정한다. 펌웨어가 다르면 `content.js` 의 `IFRAME_SELECTOR` /
  `INNER_IMG_SELECTOR` 를 고치면 된다. (아이콘 활성화·테두리 표시·실제 인식이
  모두 같은 판정을 공유한다.)
- **캡차 입력칸**: 기본값 `input[name="captcha_code"]` 은 실제 펌웨어에서 확인한
  값이다. 입력칸(`tr#captchatr1`)과 캡차 iframe(`tr#captchatr3`)이 형제 행이라
  iframe 기준 조상 탐색으로는 아이디 칸이 먼저 걸린다 — 그래서 휴리스틱에
  맡기지 않고 고정했다. 펌웨어가 다르면 이 값을 바꿔라.
- **툴바 아이콘**: 아이콘 PNG 가 없어 Chrome 기본 아이콘을 쓰는데, 활성/비활성
  회색 차이가 잘 보이지 않는다. 그래서 캡차 감지 시 초록 **`ON` 배지**를 함께
  띄운다. 배지가 확실한 신호이고 회색 여부는 보조 신호다.

## 보안 참고

- 캡차 이미지는 **사용자 본인의 solver 서버**로만 전송된다. 외부 라이브러리나
  서드파티 서비스를 호출하지 않는다.
- **로그인 폼을 자동 제출하지 않는다.** 인식 결과 확인 후 사용자가 직접
  로그인 버튼을 누른다.
- content script 는 `manifest.json` 의 `matches`(대상 로그인 페이지)에서만
  주입되고, 툴바 아이콘도 그 페이지에서 캡차 이미지가 실제로 확인된 탭에서만
  켜진다. 대상 페이지 외에서는 확장이 동작하지 않는다.
- `declarativeContent` 권한은 더 이상 필요 없어 제거했다.
