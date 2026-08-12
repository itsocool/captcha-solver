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

// 설치/업데이트 시: 액션(툴바 아이콘)을 기본 비활성화한다.
// 활성화 판정은 content.js 가 "captcha-detected" 메시지로 보고한다.
//
// declarativeContent 를 쓰지 않는 이유:
//   PageStateMatcher.css 는 복합 선택자(compound selector)만 허용해서
//   결합자(">", " ", "+", "~")를 전혀 쓸 수 없다. "body > form > img" 를
//   넣으면 "Invalid CSS selector" 로 던진다. 반면 content script 의
//   querySelector 는 전체 CSS 문법을 지원하므로, 정확히 같은 셀렉터로
//   판정한 결과를 그대로 아이콘 활성화에 쓴다.
chrome.runtime.onInstalled.addListener(() => {
  chrome.action.disable();
});

// content.js 로부터 온 solve 요청 처리:
//   dataUrl(PNG) → Blob → multipart/form-data → solver API 호출
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg) return;

  // content.js 의 셀렉터 판정 결과 → 해당 탭의 툴바 아이콘 활성/비활성.
  // 응답이 필요 없으므로 여기서 종료(return undefined)한다.
  if (msg.type === "captcha-detected") {
    const tabId = sender.tab && sender.tab.id;
    if (typeof tabId !== "number") return;

    (async () => {
      if (!msg.found) {
        await chrome.action.disable(tabId);
        await chrome.action.setBadgeText({ tabId, text: "" });
        await chrome.action.setTitle({ tabId, title: "ipTIME 캡차 인식 (캡차 없음)" });
        return;
      }

      await chrome.action.enable(tabId);

      // 전역 chrome.action.disable() 로 기본값을 끈 상태에서 탭별 enable(tabId)
      // 이 실제로 덮어썼는지 읽어서 확인한다. Chrome 문서는 "탭별 설정이 전역보다
      // 우선"이라고만 하고 이 조합을 명시하지 않는다. 덮어쓰지 못했다면 전역
      // 기본값을 해제해서라도 켠다 — 대상 페이지에서 켜지는 게 우선이고,
      // 그 대가로 다른 페이지에서도 아이콘이 회색이 아니게 된다.
      let enabled = null;
      try {
        enabled = await chrome.action.isEnabled(tabId);
      } catch (_) {
        // isEnabled 는 Chrome 110+ 다. 없으면 확인을 건너뛴다(배지로 상태 표시).
      }
      if (enabled === false) {
        await chrome.action.enable(); // 전역 기본값 해제
        await chrome.action.enable(tabId);
        console.warn(
          "[iptime-captcha] 탭별 enable 이 전역 disable 을 덮어쓰지 못해 전역 기본값을 해제했다"
        );
      }

      // 배지: 아이콘 파일이 없어 활성/비활성 회색 차이가 잘 안 보이므로,
      // 확실히 보이는 신호를 따로 얹는다. 배지는 아이콘 상태와 무관하게 그려진다.
      await chrome.action.setBadgeText({ tabId, text: "ON" });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: "#16a34a" });
      await chrome.action.setTitle({ tabId, title: "ipTIME 캡차 인식: 캡차 감지됨" });

      console.log(`[iptime-captcha] tab=${tabId} 활성화 (isEnabled=${enabled})`);
    })();

    return; // 응답 불필요
  }

  if (msg.type !== "solve") return;

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
