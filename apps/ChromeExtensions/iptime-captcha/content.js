// ipTIME Captcha Solver - content script
// 대상: http://192.168.50.4/sess-bin/login_session.cgi?noauto=1
//
// 동작: 캡차 이미지를 감지하면 canvas 로 읽어 PNG data URL 을 만들고,
// background 를 통해 solver API 로 보낸 뒤 결과를 캡차 입력칸에 채운다.
// (로그인 폼을 자동 제출하지는 않는다.)

(() => {
  "use strict";

  // 설정: 캡차 입력칸 셀렉터를 강제 지정하고 싶으면 여기에 채운다.
  // 비워두면 img 가 속한 form 안에서 비어있는 텍스트 입력칸을 휴리스틱으로 고른다.
  // (ipTIME 펌웨어마다 폼 구조가 달라 정확한 셀렉터는 사용자가 확인 후 지정 권장)
  const CAPTCHA_INPUT_SELECTOR = "";

  const IMG_SELECTOR = "body > form > img";

  // noauto=1 페이지에서만 동작
  if (!location.search.includes("noauto=1")) return;

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

  // 이미지가 로드 완료됨을 보장
  function ensureLoaded(img) {
    if (img.complete && img.naturalWidth > 0) return Promise.resolve();
    return new Promise((resolve, reject) => {
      img.addEventListener("load", () => resolve(), { once: true });
      img.addEventListener("error", () => reject(new Error("캡차 이미지 로드 실패")), { once: true });
    });
  }

  // 캡차 입력칸 찾기: 설정 우선, 없으면 form 안에서 비어있는 텍스트 입력칸
  function findInput(img) {
    if (CAPTCHA_INPUT_SELECTOR) {
      return document.querySelector(CAPTCHA_INPUT_SELECTOR);
    }
    const form = img.closest("form") || document;
    const inputs = form.querySelectorAll("input");
    for (const el of inputs) {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      // text 이거나 타입 미지정, 그리고 아직 비어있는 칸
      if ((type === "text" || !el.hasAttribute("type")) && !el.value) {
        return el;
      }
    }
    return null;
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
  async function solve(img, btn) {
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

      // 입력칸 채우기 + input 이벤트 디스패치
      const input = findInput(img);
      if (input) {
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

  // 진입점: 이미지 감지 → 버튼 주입 + 1회 자동 실행
  waitFor(IMG_SELECTOR)
    .then((img) => {
      const btn = makeButton();
      btn.addEventListener("click", () => solve(img, btn));
      solve(img, btn); // 자동 1회 실행
    })
    .catch((err) => console.warn("[iptime-captcha]", err.message));
})();
