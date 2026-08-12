// ipTIME Captcha Solver - background service worker (MV3)
//
// 설정 상수 -----------------------------------------------------------------
// 브라우저 네트워크에서 도달 가능한 captcha-solver API 베이스 주소.
//   - 기본값: 사내망 직접 접속용 (HTTP)
//   - 대안(HTTPS 프록시): "https://dev.hyperinfo.co.kr:12005"
// HTTPS 프록시로 바꾸면 manifest.json 의 host_permissions 에도 해당
// 도메인을 추가해야 fetch 가 허용된다.
const API_BASE = "http://192.168.50.98:30008";
const CAPTCHA_ID = "iptime";

// 설치/업데이트 시: 기본적으로 액션(툴바 아이콘)을 비활성화하고,
// 대상 URL + 캡차 이미지 셀렉터가 감지될 때만 활성화(ShowAction)한다.
chrome.runtime.onInstalled.addListener(() => {
  chrome.action.disable();

  chrome.declarativeContent.onPageChanged.removeRules(undefined, () => {
    chrome.declarativeContent.onPageChanged.addRules([
      {
        conditions: [
          new chrome.declarativeContent.PageStateMatcher({
            pageUrl: {
              hostEquals: "192.168.50.4",
              pathEquals: "/sess-bin/login_session.cgi",
              queryContains: "noauto=1",
            },
            // 캡차 이미지가 실제로 존재할 때만 매칭
            css: ["body > form > img"],
          }),
        ],
        actions: [new chrome.declarativeContent.ShowAction()],
      },
    ]);
  });
});

// content.js 로부터 온 solve 요청 처리:
//   dataUrl(PNG) → Blob → multipart/form-data → solver API 호출
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.type !== "solve") return;

  (async () => {
    try {
      // data URL → Blob (fetch 로 손쉽게 변환)
      const blob = await (await fetch(msg.dataUrl)).blob();

      const form = new FormData();
      form.append("captcha_id", CAPTCHA_ID);
      form.append("image", blob, "captcha.png");

      const res = await fetch(API_BASE + "/api/v1/predictImage", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        sendResponse({ ok: false, error: `HTTP ${res.status} ${res.statusText}` });
        return;
      }
      const data = await res.json();
      sendResponse({
        ok: true,
        prediction: data.prediction,
        confidence: data.confidence,
      });
    } catch (err) {
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
    }
  })();

  // 비동기 sendResponse 를 위해 true 반환
  return true;
});
