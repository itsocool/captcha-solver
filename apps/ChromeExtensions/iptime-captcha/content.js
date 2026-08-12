// ipTIME Captcha Solver - content script
// 대상: http://192.168.50.4/sess-bin/login_session.cgi
//
// 동작: 캡차 iframe 안의 이미지를 canvas 로 읽어 PNG data URL 을 만들고,
// background 를 통해 solver API 로 보낸 뒤 결과를 캡차 입력칸에 채운다.
// (로그인 폼을 자동 제출하지는 않는다.)

(() => {
  "use strict";

  // 캡차 입력칸 셀렉터. 실제 펌웨어에서 확인한 값으로 고정한다.
  //
  // name 속성으로 잡는 이유: 이 칸은 tr#captchatr1 에 있고 캡차 iframe 은
  // tr#captchatr3 에 있는 형제 행이라, iframe 기준 조상 탐색으로는 공통 조상이
  // 아이디·비밀번호 행까지 포함하는 tbody 가 되어 문서 순서상 아이디 칸이 먼저
  // 걸린다. 위치 기반 셀렉터(tr#captchatr1:nth-of-type(6) > td > input.login_input)
  // 도 행이 하나만 추가되면 깨진다. name 은 서버로 전송되는 기능적 속성이라
  // 레이아웃 변경에 영향받지 않는다.
  //
  // 빈 문자열로 두면 아래 findInput() 의 휴리스틱으로 폴백한다.
  const CAPTCHA_INPUT_SELECTOR = 'input[name="captcha_code"]';

  // 캡차는 top document 의 <img> 가 아니라 별도 iframe 안에 렌더된다.
  //   body > form > table.navi_login_table > ... > tr#captchatr3 > ...
  //     > table.login_input > tbody > tr > td > iframe#iframe_captcha
  //   (src=/sess-bin/captcha.cgi, 201x70)
  // ID 가 유일하므로 전체 경로 대신 ID 로 잡는다. 전체 경로는 테이블 중첩에
  // 의존해서 펌웨어 레이아웃이 조금만 바뀌어도 깨진다.
  const IFRAME_SELECTOR = "#iframe_captcha";

  // iframe 문서 안의 실제 캡차 이미지. captcha.cgi 가 이미지를 직접 반환하면
  // Chrome 이 생성한 image document 의 img 를, HTML 을 반환하면 그 안의 img 를
  // 잡는다. 어느 쪽이든 same-origin(192.168.50.4) 이라 canvas 가 오염되지 않는다.
  const INNER_IMG_SELECTOR = "img";

  // 활성화 시각 표시: 캡차 테두리를 녹색 점선으로.
  //
  // border 가 아니라 outline 을 쓴다. border 는 박스 크기를 키워 레이아웃을
  // 밀어내는데, 이 폼은 테이블 중첩이라 캡차 칸과 옆 칸이 같이 밀린다.
  // outline 은 리플로우를 일으키지 않는다.
  const DETECTED_OUTLINE = "2px dashed #16a34a";

  function markDetected(el) {
    el.style.outline = DETECTED_OUTLINE;
    el.style.outlineOffset = "2px";
  }

  // 셀렉터 판정 결과를 background 로 보고 → 툴바 아이콘 활성/비활성.
  // 아이콘은 표시용이라 실패해도 본 기능에 영향이 없으므로 조용히 무시한다.
  function reportDetection(found) {
    try {
      const p = chrome.runtime.sendMessage({ type: "captcha-detected", found });
      if (p && typeof p.catch === "function") p.catch(() => {});
    } catch (_) {}
  }

  // 셀렉터가 나타날 때까지 대기 (동적 렌더링 대비): MutationObserver + timeout
  function waitFor(selector, timeoutMs = 10000) {
    return new Promise((resolve, reject) => {
      const found = document.querySelector(selector);
      if (found) return resolve(found);

      const obs = new MutationObserver(() => {
        const el = document.querySelector(selector);
        if (el) {
          obs.disconnect();
          clearTimeout(timer);
          resolve(el);
        }
      });
      obs.observe(document.documentElement, { childList: true, subtree: true });

      const timer = setTimeout(() => {
        obs.disconnect();
        reject(new Error(`timeout: ${selector} 미검출`));
      }, timeoutMs);
    });
  }

  // iframe 내부 문서 접근. same-origin 이지만 아직 로드 전이면 null 이 나온다.
  function innerDoc(iframeEl) {
    try {
      return iframeEl.contentDocument;
    } catch (_) {
      return null; // cross-origin 으로 바뀐 경우 방어
    }
  }

  // iframe 안의 캡차 이미지를 기다린다.
  // MutationObserver 는 아직 존재하지 않는 문서에 붙일 수 없어서(로드 전에는
  // contentDocument 가 null) load 이벤트 + 폴링을 함께 쓴다. 캡차 새로고침으로
  // iframe 이 다시 로드되는 경우도 폴링이 받아준다.
  function waitForInnerImg(iframeEl, timeoutMs = 10000) {
    return new Promise((resolve, reject) => {
      const deadline = Date.now() + timeoutMs;
      let timer = null;

      const tick = () => {
        const doc = innerDoc(iframeEl);
        const img = doc && doc.querySelector(INNER_IMG_SELECTOR);
        if (img) {
          clearTimeout(timer);
          return resolve(img);
        }
        if (Date.now() >= deadline) {
          return reject(
            new Error(`timeout: ${IFRAME_SELECTOR} 내부 ${INNER_IMG_SELECTOR} 미검출`)
          );
        }
        timer = setTimeout(tick, 200);
      };

      iframeEl.addEventListener("load", tick, { once: true });
      tick();
    });
  }

  // 이미지가 로드 완료됨을 보장
  function ensureLoaded(img) {
    if (img.complete && img.naturalWidth > 0) return Promise.resolve();
    return new Promise((resolve, reject) => {
      img.addEventListener("load", () => resolve(), { once: true });
      img.addEventListener("error", () => reject(new Error("캡차 이미지 로드 실패")), { once: true });
    });
  }

  // 캡차 입력칸 찾기: 설정 우선, 없으면 캡차 iframe 에서 조상을 타고 바깥으로
  // 넓혀가며 가장 가까운 빈 텍스트 입력칸을 고른다.
  //
  // 주의: 예전처럼 form 전체를 한 번에 훑으면 안 된다. 로그인 폼에는 아이디 칸이
  // 함께 있고 그쪽도 type=text + 빈 값이라, 문서 순서상 먼저 걸려서 캡차 값이
  // 아이디 칸에 들어간다. 캡차 입력칸은 iframe 과 같은 테이블 안에 있으므로
  // 가까운 조상부터 찾으면 그쪽이 먼저 잡힌다.
  function findInput(anchor) {
    if (CAPTCHA_INPUT_SELECTOR) {
      return document.querySelector(CAPTCHA_INPUT_SELECTOR);
    }

    const isCandidate = (el) => {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      return (type === "text" || !el.hasAttribute("type")) && !el.value;
    };

    for (let node = anchor.parentElement; node; node = node.parentElement) {
      for (const el of node.querySelectorAll("input")) {
        if (isCandidate(el)) return el;
      }
      if (node.tagName === "FORM") break; // 폼 밖까지 나가지 않는다
    }
    return null;
  }

  // 입력칸의 안내 문구 제거.
  //
  // 이 칸은 placeholder 가 아니라 배경 이미지로 안내 문구를 그린다:
  //   background-image: url("/images2/login_str_captcha_bg.kr.gif")
  // 그리고 페이지가 onfocus="ChangeCaptchaInputBg('clear')" 로 지운다.
  // 값만 프로그램으로 넣으면 포커스가 없어 문구가 남고 입력 텍스트와 겹친다.
  //
  // focus() 를 실제로 호출하면 인라인 onfocus 핸들러가 페이지 월드에서 정상
  // 실행돼 페이지 자신의 로직으로 지워진다 (content script 는 isolated world 라
  // ChangeCaptchaInputBg 를 직접 부를 수 없다). 핸들러가 없는 펌웨어를 대비해
  // 배경 이미지도 직접 지운다.
  //
  // 포커스가 캡차 칸에 남는 건 의도한 것이다. 이 칸의 onkeydown 이 Enter 로
  // LoginProcess() 를 호출하므로, 값 확인 후 바로 Enter 를 칠 수 있다.
  function clearInputPrompt(input) {
    try {
      input.focus();
    } catch (_) {}
    if (input.style.backgroundImage && input.style.backgroundImage !== "none") {
      input.style.backgroundImage = "none";
    }
  }

  // 플로팅 버튼 (수동 재실행 + 결과 표시용)
  function makeButton() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "🔓 캡차 자동 인식";
    Object.assign(btn.style, {
      position: "fixed",
      top: "12px",
      right: "12px",
      zIndex: "2147483647",
      padding: "8px 12px",
      fontSize: "13px",
      fontFamily: "sans-serif",
      color: "#fff",
      background: "#2563eb",
      border: "none",
      borderRadius: "6px",
      boxShadow: "0 2px 6px rgba(0,0,0,.3)",
      cursor: "pointer",
    });
    document.body.appendChild(btn);
    return btn;
  }

  function setStatus(btn, text, ok) {
    btn.textContent = text;
    btn.style.background = ok === true ? "#16a34a" : ok === false ? "#dc2626" : "#2563eb";
  }

  // 핵심: 캡차 이미지 → canvas → PNG → solver → 입력칸 채우기
  //   img      : iframe 내부의 실제 캡차 이미지
  //   anchor   : top document 의 iframe 엘리먼트 (입력칸 탐색 기준점)
  async function solve(img, anchor, btn) {
    try {
      setStatus(btn, "⏳ 인식 중…", null);
      await ensureLoaded(img);

      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      const dataUrl = canvas.toDataURL("image/png");

      const resp = await chrome.runtime.sendMessage({ type: "solve", dataUrl });
      if (!resp || !resp.ok) {
        setStatus(btn, `❌ ${resp && resp.error ? resp.error : "오류"}`, false);
        return;
      }

      const { prediction, confidence } = resp;

      // 안내 문구 제거 → 값 채우기 → input 이벤트 디스패치 (순서 중요)
      const input = findInput(anchor);
      if (input) {
        clearInputPrompt(input);
        input.value = prediction;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }

      // 클립보드 복사 (실패해도 무시)
      try {
        await navigator.clipboard.writeText(prediction);
      } catch (_) {}

      const pct = typeof confidence === "number" ? ` (${Math.round(confidence * 100)}%)` : "";
      setStatus(btn, `✅ ${prediction}${pct}${input ? "" : " · 입력칸 못찾음"}`, true);
    } catch (err) {
      setStatus(btn, `❌ ${err.message || err}`, false);
    }
  }

  // 진입점: 캡차 iframe 감지 → 내부 이미지 확보 → 버튼 주입 + 1회 자동 실행
  //
  // 예전에는 여기서 location.search 의 noauto=1 을 요구했지만, 실제 로그인
  // 페이지는 쿼리 없이 열려서 항상 early-return 되고 있었다. 대상 판정은
  // manifest 의 matches(login_session.cgi) + 캡차 iframe 존재 여부로 충분하다.
  waitFor(IFRAME_SELECTOR)
    .then((iframeEl) => waitForInnerImg(iframeEl).then((img) => ({ iframeEl, img })))
    .then(({ iframeEl, img }) => {
      reportDetection(true);
      markDetected(iframeEl);
      const btn = makeButton();
      btn.addEventListener("click", () => solve(img, iframeEl, btn));
      solve(img, iframeEl, btn); // 자동 1회 실행
    })
    .catch((err) => {
      reportDetection(false);
      console.warn("[iptime-captcha]", err.message);
    });
})();
