// Chat widget toggle behaviour (moved from template)
(function () {
  document.addEventListener('DOMContentLoaded', function () {
    var toggleBtn = document.getElementById('chatToggleBtn');
    var widget = document.getElementById('chatWidget');
    var closeBtn = document.getElementById('chatCloseBtn');
    var sendBtn = document.getElementById('chatSendBtn');
    var input = document.getElementById('chatInput');
    var chatFooter = document.getElementById('chatFooter');
    var chatFileInput = document.getElementById('chatFileInput');
    var chatFileBtn = document.getElementById('chatFileBtn');
    var chatFilePreview = document.getElementById('chatFilePreview');
    var chatFileName = document.getElementById('chatFileName');
    var chatFileRemove = document.getElementById('chatFileRemove');

    if (!toggleBtn || !widget) return;

    // File attachment state
    var attachedFile = null;

    // Theme handling: widget follows global theme only. It will set
    // data-theme on the widget based on document-level indicators
    // (data-theme on <html> or class 'dark') or prefers-color-scheme.

    function applyTheme(theme) {
      if (!theme) {
        widget.removeAttribute('data-theme');
        return;
      }
      widget.setAttribute('data-theme', theme);
    }

    function detectGlobalTheme() {
      var html = document.documentElement;
      var body = document.body;
      var attr = html.getAttribute('data-theme') || body.getAttribute('data-theme');
      if (attr) return attr === 'dark' || attr === 'Dark' ? 'dark' : 'light';
      if (html.classList.contains('dark') || body.classList.contains('dark')) return 'dark';
      if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
      return 'light';
    }

    // Initialize widget theme to follow global theme
    applyTheme(detectGlobalTheme());

    // Observe global theme changes and update widget accordingly
    var observer = new MutationObserver(function () {
      applyTheme(detectGlobalTheme());
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class','data-theme'] });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class','data-theme'] });
    if (window.matchMedia) {
      try {
        var mq = window.matchMedia('(prefers-color-scheme: dark)');
        mq.addEventListener && mq.addEventListener('change', function () { applyTheme(detectGlobalTheme()); });
      } catch (e) { /* ignore */ }
    }

    // Basic open/close behaviour
    function openWidget() {
      widget.classList.add('open');
      widget.setAttribute('aria-hidden', 'false');
      toggleBtn.setAttribute('aria-expanded', 'true');
      toggleBtn.title = '챗봇 닫기';
      if (input) input.focus();
    }

    function closeWidget() {
      widget.classList.remove('open');
      widget.setAttribute('aria-hidden', 'true');
      toggleBtn.setAttribute('aria-expanded', 'false');
      toggleBtn.title = '챗봇 열기';
      toggleBtn.focus();
    }

    toggleBtn.addEventListener('click', function (e) {
      if (widget.classList.contains('open')) closeWidget(); else openWidget();
    });

    if (closeBtn) closeBtn.addEventListener('click', closeWidget);

    // File attachment handlers
    if (chatFileBtn && chatFileInput) {
      chatFileBtn.addEventListener('click', function() {
        chatFileInput.click();
      });

      chatFileInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
          handleFileSelect(e.target.files[0]);
        }
      });
    }

    if (chatFileRemove) {
      chatFileRemove.addEventListener('click', function() {
        clearAttachment();
      });
    }

    // Drag and drop on chat footer
    if (chatFooter) {
      chatFooter.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        chatFooter.classList.add('drag-over');
      });

      chatFooter.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        chatFooter.classList.remove('drag-over');
      });

      chatFooter.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        chatFooter.classList.remove('drag-over');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          handleFileSelect(e.dataTransfer.files[0]);
        }
      });
    }

    function handleFileSelect(file) {
      if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 첨부할 수 있습니다.');
        return;
      }
      
      attachedFile = file;
      if (chatFileName) {
        chatFileName.textContent = file.name;
      }
      if (chatFilePreview) {
        chatFilePreview.classList.remove('d-none');
      }
    }

    function clearAttachment() {
      attachedFile = null;
      if (chatFileInput) {
        chatFileInput.value = '';
      }
      if (chatFilePreview) {
        chatFilePreview.classList.add('d-none');
      }
    }

    // Close on ESC
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && widget.classList.contains('open')) {
        closeWidget();
      }
    });

    // Send behaviour — GET message to /api/v1/chat
    if (sendBtn && input) {
      // auto-resize textarea up to 4 lines, then enable internal scroll
      function adjustTextareaHeight(el) {
        if (!el) return;
        el.style.height = 'auto';
        var cs = window.getComputedStyle(el);
        var lineHeight = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.2;
        var paddingTop = parseFloat(cs.paddingTop) || 0;
        var paddingBottom = parseFloat(cs.paddingBottom) || 0;
        var minHeight = lineHeight + paddingTop + paddingBottom + 4; // ~1 line
        var maxHeight = lineHeight * 4 + paddingTop + paddingBottom + 4; // up to 4 lines
        // compute desired height based on content, but enforce min/max
        var contentHeight = el.scrollHeight || minHeight;
        var newHeight = Math.min(Math.max(contentHeight, minHeight), maxHeight);
        el.style.height = newHeight + 'px';
        el.style.overflowY = (contentHeight > maxHeight) ? 'auto' : 'hidden';
      }

      // initialize height
      adjustTextareaHeight(input);
      input.addEventListener('input', function () { adjustTextareaHeight(input); });
      sendBtn.addEventListener('click', async function () {
        var v = input.value && input.value.trim();
        if (!v && !attachedFile) return;

        var bodyEl = widget.querySelector('.chat-body');
        // show user message as bubble
        function appendMessage(role, text) {
          var msg = document.createElement('div');
          msg.className = 'chat-message ' + role;

          var avatar = document.createElement('div');
          avatar.className = 'chat-avatar';
          avatar.setAttribute('aria-hidden', 'true');

          var bubble = document.createElement('div');
          bubble.className = 'chat-bubble';
          bubble.setAttribute('role', 'article');
          bubble.setAttribute('aria-label', (role === 'user' ? '사용자 메시지' : '봇 응답'));
          bubble.textContent = text;

          if (role === 'user') {
            // user: avatar then bubble (avatar on left)
            avatar.innerHTML = '<i class="bi bi-person-fill" aria-hidden="true"></i>';
            msg.appendChild(avatar);
            msg.appendChild(bubble);
          } else {
            // bot: bubble then avatar (avatar on right)
            avatar.innerHTML = '<i class="bi bi-robot" aria-hidden="true"></i>';
            msg.appendChild(bubble);
            msg.appendChild(avatar);
          }

          bodyEl.appendChild(msg);
          bodyEl.scrollTop = bodyEl.scrollHeight;
        }

        var userMsg = v || '';
        if (attachedFile) {
          userMsg += (userMsg ? '\n' : '') + '📎 ' + attachedFile.name;
        }
        appendMessage('user', userMsg);

        // disable input while waiting and show spinner
        sendBtn.disabled = true;
        input.disabled = true;
        try {
          var iconEl = sendBtn.querySelector('.send-icon');
          var spinnerEl = sendBtn.querySelector('.send-spinner');
          if (iconEl) iconEl.classList.add('visually-hidden');
          if (spinnerEl) spinnerEl.classList.remove('visually-hidden');
          sendBtn.setAttribute('aria-busy', 'true');
        } catch (e) { /* ignore */ }

        try {
          // include stream flag if the toggle exists
          var streamToggle = document.getElementById('chatStreamToggle');
          var useStream = false;
          if (streamToggle) {
            useStream = !!streamToggle.checked;
          }

          // Prepare request based on whether file is attached
          var fetchOptions = { method: 'GET' };
          var url = new URL('/api/v1/chat', window.location.origin);
          
          if (attachedFile) {
            // Use POST with FormData for file upload
            var formData = new FormData();
            formData.append('message', v || '');
            formData.append('image', attachedFile);
            formData.append('stream', useStream ? '1' : '0');
            
            url = new URL('/api/v1/chat', window.location.origin);
            fetchOptions = {
              method: 'POST',
              body: formData
            };
          } else {
            // Use GET with query params for text-only
            url.searchParams.append('message', v);
            url.searchParams.append('stream', useStream ? '1' : '0');
          }

          if (useStream) {
            // open a streaming fetch and read chunks progressively
            var resp = await fetch(url, fetchOptions);
            if (!resp.ok) {
              var dataErr = null;
              try { dataErr = await resp.json(); } catch (e) { /* ignore */ }
              var errMsg = (dataErr && (dataErr.error || dataErr.message)) || resp.statusText || 'Unknown error';
              appendMessage('bot', '오류: ' + errMsg);
            } else if (!resp.body) {
              appendMessage('bot', '스트리밍을 지원하지 않는 응답입니다.');
            } else {
              // create a bot bubble and progressively append
              var botMsgEl = document.createElement('div');
              botMsgEl.className = 'chat-message bot';
              var avatar = document.createElement('div'); avatar.className = 'chat-avatar'; avatar.setAttribute('aria-hidden','true'); avatar.innerHTML = '<i class="bi bi-robot" aria-hidden="true"></i>';
              var bubble = document.createElement('div'); bubble.className = 'chat-bubble'; bubble.setAttribute('role','article'); bubble.setAttribute('aria-label','봇 응답');
              bubble.textContent = '';
              botMsgEl.appendChild(bubble);
              botMsgEl.appendChild(avatar);
              widget.querySelector('.chat-body').appendChild(botMsgEl);
              widget.querySelector('.chat-body').scrollTop = widget.querySelector('.chat-body').scrollHeight;

              const reader = resp.body.getReader();
              const decoder = new TextDecoder();
              let done = false;
              
              // Streaming mode expects plain text chunks only
              while (!done) {
                const rr = await reader.read();
                done = rr.done;
                if (rr.value) {
                  const chunk = decoder.decode(rr.value, { stream: true });
                  
                  // Simply append plain text chunks directly
                  if (chunk && chunk.trim()) {
                    bubble.textContent += chunk;
                    widget.querySelector('.chat-body').scrollTop = widget.querySelector('.chat-body').scrollHeight;
                  }
                }
              }
            }
          } else {
            var resp = await fetch(url, fetchOptions);
            var data = await resp.json();
            if (!resp.ok) {
              var errMsg = data.error || resp.statusText || 'Unknown error';
              appendMessage('bot', '오류: ' + errMsg);
            } else {
              var text = data.text || data.response || data.predicted || '';
              appendMessage('bot', text);
            }
          }
        } catch (err) {
          appendMessage('bot', '요청 실패: ' + (err.message || err));
        } finally {
          // hide spinner and re-enable
          try {
            var iconEl2 = sendBtn.querySelector('.send-icon');
            var spinnerEl2 = sendBtn.querySelector('.send-spinner');
            if (iconEl2) iconEl2.classList.remove('visually-hidden');
            if (spinnerEl2) spinnerEl2.classList.add('visually-hidden');
            sendBtn.removeAttribute('aria-busy');
          } catch (e) { /* ignore */ }
          sendBtn.disabled = false;
          input.disabled = false;
          input.value = '';
          clearAttachment();
          input.focus();
        }
      });

      // Support Enter to send, Shift+Enter to insert newline
      input.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') {
          if (ev.shiftKey) {
            // allow newline
            return;
          }
          // send on plain Enter
          ev.preventDefault();
          sendBtn.click();
        }
      });
    }
  });
})();
