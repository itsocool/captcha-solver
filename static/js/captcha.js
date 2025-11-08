const imageInput = document.getElementById('image');
const preview = document.getElementById('preview');
const dropzone = document.getElementById('dropzone');
const dropFilename = document.getElementById('dropFilename');
const submitBtn = document.getElementById('submitBtn');
const ocrBtn = document.getElementById('ocrBtn');
const clearBtn = document.getElementById('clearBtn');
const spinner = document.getElementById('spinner');
const alerts = document.getElementById('alerts');

function showAlert(message, type = 'info') {
  alerts.innerHTML = `<div class="alert alert-${type} alert-dismissible" role="alert">
      <div class="monospace">${message.replace(/\n/g, '<br>')}</div>
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>`;
}

// 추가: 이미지 선택 여부에 따라 [예측 실행] 버튼 활성/비활성 처리
function updateSubmitButtonState() {
  const hasFile = imageInput.files && imageInput.files.length > 0;
  submitBtn.disabled = !hasFile;
}

// 초기 상태 반영
updateSubmitButtonState();

// 단일 모드(캡챠 전용). 엔드포인트는 고정입니다.

imageInput.addEventListener('change', (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    preview.classList.add('d-none');
    preview.src = '';
    dropFilename.textContent = '';
    updateSubmitButtonState(); // 변경: 선택 해제 시 버튼 비활성화
    return;
  }
  const url = URL.createObjectURL(f);
  preview.src = url;
  preview.classList.remove('d-none');
  dropFilename.textContent = f.name;
  updateSubmitButtonState(); // 변경: 파일 선택 시 버튼 활성화
});

// Dropzone handlers
dropzone.addEventListener('click', () => imageInput.click());
['dragenter', 'dragover'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop', 'dragend'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', (e) => {
  e.preventDefault(); e.stopPropagation();
  const dt = e.dataTransfer;
  if (!dt || !dt.files || dt.files.length === 0) return;
  const f = dt.files[0];
  // populate the hidden file input so existing code can use it
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(f);
  imageInput.files = dataTransfer.files;
  // trigger change
  const ev = new Event('change', { bubbles: true });
  imageInput.dispatchEvent(ev);
});

function clearForm() {
  imageInput.value = '';
  preview.src = '';
  preview.classList.add('d-none');
  alerts.innerHTML = '';
  if (typeof dropFilename !== 'undefined') dropFilename.textContent = '';
  updateSubmitButtonState(); // 변경: 초기화 후 버튼 비활성화
}

ocrBtn.addEventListener('click', async () => {
      alerts.innerHTML = '';
      const f = imageInput.files && imageInput.files[0];
      if (!f) { showAlert('이미지를 선택하세요.', 'warning'); return; }

      const form = new FormData();
      form.append('image', f, f.name);

      spinner.classList.remove('d-none');
      ocrBtn.disabled = true;
      clearBtn.disabled = true;

      try {
        const resp = await fetch('/api/v1/ocr', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) {
          showAlert(`오류: ${data.error || resp.statusText}`, 'danger');
        } else {
          // Add a result card with image, predicted value, confidence and processing time (ms)
          addResultCard(f, data.predicted, data.confidence ?? 'N/A', data.processing_ms ?? null, data.bbox ?? null);
          // 자동 초기화: 결과를 리스트에 추가한 뒤 입력폼을 초기화
          clearForm();
        }
      } catch (err) {
        showAlert('요청 실패: ' + err.message, 'danger');
      } finally {
        spinner.classList.add('d-none');
        ocrBtn.disabled = false;
        clearBtn.disabled = false;
      }
    });

clearBtn.addEventListener('click', () => {
  clearForm();
});

submitBtn.addEventListener('click', async () => {
  alerts.innerHTML = '';
  const f = imageInput.files && imageInput.files[0];
  if (!f) { showAlert('이미지를 선택하세요.', 'warning'); return; }

  const form = new FormData();
  form.append('image', f, f.name);

  spinner.classList.remove('d-none');
  submitBtn.disabled = true;
  clearBtn.disabled = true;

  try {
    // 캡챠 전용 엔드포인트
    const endpoint = '/api/v1/captcha';
    const resp = await fetch(endpoint, { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) {
      showAlert(`오류: ${data.error || resp.statusText}`, 'danger');
    } else {
      // Add a result card with image, predicted value, confidence and processing time (ms)
      addResultCard(f, data.predicted, data.confidence ?? 'N/A', data.processing_ms ?? null);
      // 자동 초기화: 결과를 리스트에 추가한 뒤 입력폼을 초기화
      clearForm();
    }
  } catch (err) {
    showAlert('요청 실패: ' + err.message, 'danger');
  } finally {
    spinner.classList.add('d-none');
    submitBtn.disabled = false;
    clearBtn.disabled = false;
  }
});

function formatConfidence(c) {
  if (c === null || c === undefined) return 'N/A';
  const num = Number(c);
  if (isNaN(num)) return String(c);
  // if in [0,1], show percent
  if (num <= 1) return (num * 100).toFixed(2) + '%';
  // else assume already percentage-like
  return num.toFixed(2) + (num > 10 ? '%' : '%');
}

function addResultCard(file, predicted, confidence, processingMs, bbox) {
  const list = document.getElementById('resultsList');
  const url = preview.src || (file ? URL.createObjectURL(file) : '');
  const time = new Date().toLocaleString();
  const confText = formatConfidence(confidence);
  const div = document.createElement('div');
  div.className = 'card mb-3 result-card';
  
  // Canvas wrapper for bbox visualization
  const canvasId = `canvas-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const imageContainerId = `img-container-${canvasId}`;
  
  div.innerHTML = `
        <div class="row g-0 align-items-center">
          <div class="col-auto p-2">
            <div id="${imageContainerId}" style="position:relative; width:140px; height:80px;">
              <canvas id="${canvasId}" style="position:absolute; top:0; left:0; width:100%; height:100%;"></canvas>
            </div>
          </div>
          <div class="col">
            <div class="card-body py-2">
                <h6 class="card-title mb-1">예측값: <span class="predicted-text fw-bold monospace">${predicted}</span></h6>
                <p class="card-text mb-1">컨피던스: <span class="badge confidence-badge bg-teal-soft text-teal">${confText}</span></p>
                <p class="card-text mb-1">처리시간: <span class="badge confidence-badge bg-teal-soft text-teal">${processingMs}ms</span></p>
            </div>
          </div>
          <div class="col-auto pe-2">
            <button class="btn btn-outline-secondary btn-sm copy-btn" title="Copy prediction"><i class="bi bi-clipboard"></i></button>
            <button class="btn btn-outline-danger btn-sm ms-1 remove-btn" title="Remove"><i class="bi bi-trash"></i></button>
          </div>
        </div>
      `;

  // copy handler
  div.querySelector('.copy-btn').addEventListener('click', () => {
    navigator.clipboard && navigator.clipboard.writeText(predicted);
  });
  // remove handler
  div.querySelector('.remove-btn').addEventListener('click', () => {
    div.remove();
  });

  // prepend newest on top
  list.prepend(div);
  
  // Draw image and bounding boxes on canvas
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const img = new Image();
  
  img.onload = () => {
    // Set canvas size to match display size
    const displayWidth = 140;
    const displayHeight = 80;
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    // Calculate scaling to fit image within canvas (object-fit: contain)
    const imgAspect = img.width / img.height;
    const canvasAspect = displayWidth / displayHeight;
    let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
    
    if (imgAspect > canvasAspect) {
      drawWidth = displayWidth;
      drawHeight = displayWidth / imgAspect;
      offsetY = (displayHeight - drawHeight) / 2;
    } else {
      drawHeight = displayHeight;
      drawWidth = displayHeight * imgAspect;
      offsetX = (displayWidth - drawWidth) / 2;
    }
    
    // Draw image
    ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
    
    // Draw bounding boxes if available
    if (bbox && Array.isArray(bbox) && bbox.length > 0) {
      const scaleX = drawWidth / img.width;
      const scaleY = drawHeight / img.height;
      
      ctx.strokeStyle = '#00ff00'; // 녹색
      ctx.lineWidth = 2;
      
      bbox.forEach(box => {
        // bbox format: [x, y, width, height] 또는 [x1, y1, x2, y2]
        if (box.length >= 4) {
          const x1 = box[0] * scaleX + offsetX;
          const y1 = box[1] * scaleY + offsetY;
          
          // Check if it's [x, y, w, h] or [x1, y1, x2, y2]
          let width, height;
          if (box[2] < img.width && box[3] < img.height && box[2] > 0 && box[3] > 0) {
            // Likely width/height format
            width = box[2] * scaleX;
            height = box[3] * scaleY;
          } else {
            // Likely x2/y2 format
            width = (box[2] - box[0]) * scaleX;
            height = (box[3] - box[1]) * scaleY;
          }
          
          ctx.strokeRect(x1, y1, width, height);
        }
      });
    }
  };
  
  img.src = url;
}

// Theme handling: light / dark / auto
const THEME_KEY = 'theme-mode'; // values: 'light' | 'dark' | 'auto'
const themeBtn = document.getElementById('themeBtn');
const themeIcon = document.getElementById('themeIcon');

function applyTheme(mode) {
  if (mode === 'auto') {
    // follow system
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
  } else {
    document.body.setAttribute('data-theme', mode);
  }
  updateIcon(mode);
}

function updateIcon(mode) {
  themeIcon.className = '';
  if (mode === 'light') themeIcon.classList.add('bi', 'bi-sun-fill');
  else if (mode === 'dark') themeIcon.classList.add('bi', 'bi-moon-fill');
  else themeIcon.classList.add('bi', 'bi-circle-half');
}

function cycleTheme() {
  const cur = localStorage.getItem(THEME_KEY) || 'auto';
  const order = ['light', 'dark', 'auto'];
  let next = order[(order.indexOf(cur) + 1) % order.length];
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

// initialize theme on load
(function () {
  const saved = localStorage.getItem(THEME_KEY) || 'auto';
  applyTheme(saved);
  // if auto, listen to system changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      const mode = localStorage.getItem(THEME_KEY) || 'auto';
      if (mode === 'auto') applyTheme('auto');
    });
  }
})();

// ensure button state remains correct after theme init
updateSubmitButtonState();

themeBtn.addEventListener('click', cycleTheme);

// Server health check
async function checkServerHealth(port) {
  const indicator = document.getElementById(`status-${port}`);
  try {
    const response = await fetch(`https://dev.hyperinfo.co.kr/health/${port}`, {
      method: 'GET',
      mode: 'cors',
      cache: 'no-cache',
      signal: AbortSignal.timeout(3000)
    });

    if (response.ok) {
      indicator.className = 'status-indicator status-online';
    } else {
      indicator.className = 'status-indicator status-offline';
    }
  } catch (error) {
    indicator.className = 'status-indicator status-offline';
  }
}

// Check both servers every 5 seconds
function startHealthChecks() {
  checkServerHealth(5000);
  checkServerHealth(5001);
  setInterval(() => {
    checkServerHealth(5000);
    checkServerHealth(5001);
  }, 5000);
}

// Start health checks when page loads
startHealthChecks();
