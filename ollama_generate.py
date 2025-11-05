from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.parse import urlparse
from ollama import ChatResponse, chat
import requests

def extract_message_content(resp_json: dict[str, Any]) -> str | None:
    """Best-effort extraction of the textual response from various Ollama response shapes."""
    # common shapes: {'message': {'content': '...'}} or {'message': {'role':..., 'content': '...'}}
    if not isinstance(resp_json, dict):
        return None
    msg = resp_json.get("message")
    if isinstance(msg, dict):
        cont = msg.get("content")
        if isinstance(cont, str):
            return cont
    # fallback common key names
    for key in ("text", "output", "result", "completion", "body"):
        v = resp_json.get(key)
        if isinstance(v, str):
            return v
    # if choices list exists
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            for k in ("text", "message", "output"):
                vv = c0.get(k)
                if isinstance(vv, str):
                    return vv
    return None

def generate(
    prompt: str,
    model: str = "qwen3-vl:235b-instruct-cloud",
    api_key: str | None = None,
    host: str = "https://ollama.com/api/generate",
    timeout: int = 30,
) -> str:
    """Send a single generate request to Ollama and return the extracted text.

    Raises requests.HTTPError on non-2xx responses.
    """
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": False}

    resp = requests.post(host, json=payload, headers=headers, timeout=timeout)
    # raise for HTTP errors (4xx/5xx)
    resp.raise_for_status()

    # attempt to parse JSON and extract content
    try:
        data = resp.json()
    except ValueError:
        # not JSON (unexpected) — return raw text
        return resp.text.strip()

    text = extract_message_content(data)
    if text is None:
        # return pretty-printed JSON as fallback
        return json.dumps(data, ensure_ascii=False, indent=2)
    return text

def generate_stream(
    prompt: str,
    model: str = "qwen3-vl:235b-instruct-cloud",
    api_key: str | None = None,
    host: str = "https://ollama.com/api/generate",
    timeout: int = 60,
):
    """Stream the response from the generate endpoint (best-effort).

    This will yield decoded lines as they arrive. The exact streaming format depends on
    the server (newline-separated JSON parts is common)."""
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": True}

    with requests.post(host, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            # try to parse JSON chunk
            try:
                part = json.loads(raw)
            except Exception:
                yield raw
                continue
            # extract content if present
            text = extract_message_content(part)
            if text:
                yield text
            else:
                yield json.dumps(part, ensure_ascii=False)

def _probe_models(host: str, api_key: str | None = None) -> list[str]:
    """Try to query the /models endpoint on a set of candidate base paths and return model ids found.

    This is a best-effort helper used for host/model validation.
    """
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # derive candidate bases from the provided host URL
    parsed = urlparse(host)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [f"{base}/models", f"{base}/api/models", f"{base}/models/list"]
    results: list[str] = []
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                # expect a list or dict with 'models' etc
                if isinstance(data, list):
                    results.extend([str(x) for x in data])
                elif isinstance(data, dict):
                    # common shapes: {'models': [...]} or {'items': [...]} or {'model': ...}
                    for k in ("models", "items", "model"):
                        if k in data and isinstance(data[k], list):
                            results.extend([str(x) for x in data[k]])
                    # sometimes API returns dict of model->meta
                    results.extend([str(k) for k in data.keys()])
                if results:
                    return results
        except Exception:
            continue
    return results

def ocr_image(
    image_path: str,
    model_id: str = "qwen3-vl:235b-instruct-cloud",
    api_key: str | None = None,
) -> str:
    """이미지 바이트를 받아 OCR(텍스트 추출) 결과를 반환합니다.

    동작: 이미지 바이트를 base64로 인코딩해 data URI로 프롬프트에 포함시킨 뒤
    generate()를 통해 모델에 요청합니다. 모델 응답에서 텍스트만 반환하도록 지시합니다.
    """
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass api_key")

    # base64 encode (ascii 문자열)
    # b64 = base64.b64encode(image_bytes).decode("ascii")
    
    messages=[{
    'role': 'user',
    'content': "다음 이미지에서 인식되는 텍스트만 정확히 추출하여 출력하십시오. " +
                "추가 설명이나 주석은 하지 마십시오. 이미지에 텍스트가 없다면 빈 문자열을 반환하십시오.\n\n",
    'images': [image_path]}]
    
    response: ChatResponse = chat(
        model=model_id,
        messages=messages,
        options={'temperature': 0.0},
    )
    
    return response

def cloud_chat(
    message: str,
    model_id: str = "qwen3-vl:235b-instruct-cloud",
    api_key: str | None = None,
    stream: bool = False,
) -> str:
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass api_key")

    # base64 encode (ascii 문자열)
    # b64 = base64.b64encode(image_bytes).decode("ascii")
    
    messages=[{
    'role': 'user',
    'content': message,
    }]
    
    response: ChatResponse = chat(
        model=model_id,
        messages=messages,
        options={'temperature': 0.0},
    )
    
    return response

def chat_stream(
    prompt: str,
    model: str = "qwen3-vl:235b-instruct-cloud",
    api_key: str | None = None,
    host: str = "https://ollama.com/api/generate",
    timeout: int = 60,
):
    """Stream the response from the generate endpoint (best-effort).

    This will yield decoded lines as they arrive. The exact streaming format depends on
    the server (newline-separated JSON parts is common)."""
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": True}

    with requests.post(host, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            # try to parse JSON chunk
            try:
                part = json.loads(raw)
            except Exception:
                yield raw
                continue
            # extract content if present
            text = extract_message_content(part)
            if text:
                yield text
            else:
                yield json.dumps(part, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send a generate request to an Ollama endpoint.")
    parser.add_argument("--prompt", "-p", default='삼국지 인물중 조운의 고향은 현재 중국의 어디인가?', help="Prompt text to send")
    parser.add_argument("--model", "-m", default="qwen3-vl:235b-instruct-cloud", help="Model id")
    parser.add_argument("--api-key", "-k", help="API key. If omitted, reads OLLAMA_API_KEY env var.")
    parser.add_argument(
        "--host",
        "-H",
        default=os.environ.get("OLLAMA_HOST", "https://ollama.com/api/generate"),
        help="Full generate endpoint URL (default from OLLAMA_HOST or https://ollama.com/api/generate)",
    )
    parser.add_argument("--stream", action="store_true", help="Stream the response (if server supports it)")
    parser.add_argument("--check-only", action="store_true", help="Only probe host/models and print available models")
    args = parser.parse_args(argv)
    # If host looks like a full generate URL (e.g. ends with /generate or /chat), keep it; otherwise
    # allow probing to derive base endpoints.
    try:
        # Probe models if requested or to validate model existence
        models = _probe_models(args.host, api_key=args.api_key)
        if args.check_only:
            if models:
                print("Available models:")
                for m in models:
                    print(" -", m)
            else:
                print("No models discovered at host. (network/problem or no models)")
            return 0

        if models and args.model not in models:
            print(f"Warning: model '{args.model}' not found on host. Available models (sample):")
            for m in models[:50]:
                print(" -", m)

        if args.stream:
            for chunk in generate_stream(prompt=args.prompt, model=args.model, api_key=args.api_key, host=args.host):
                print(chunk)
            return 0
        else:
            out = generate(prompt=args.prompt, model=args.model, api_key=args.api_key, host=args.host)
            print(out)
            return 0
    except requests.HTTPError as e:
        print(f"HTTP error: {e} (status {getattr(e.response, 'status_code', 'unknown')})", file=sys.stderr)
        try:
            print("Response body:", e.response.text, file=sys.stderr)
        except Exception:
            pass
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
