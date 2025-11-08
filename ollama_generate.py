from __future__ import annotations

import os
import sys
import argparse
import requests
import json
import time
from ollama import ChatResponse, chat

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'https://ollama.com/api')
os.environ.setdefault('OLLAMA_HOST', 'https://ollama.com/api')
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen3-vl:235b-instruct-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "f306bbcebc3642f39e43744afa3c13b7.aH1Gn9Al9C-LvxKczozDTU8s")

def ocr_image(
    image_path: str = None,
    model_id: str = OLLAMA_MODEL_ID,
    api_key: str = OLLAMA_API_KEY,
) -> dict:
    """OCR 이미지 처리 및 결과 반환
    
    Returns:
        dict: {
            "text": str,
            "confidence": float,
            "processing_ms": int,
            "bbox": list (optional)
        }
    """
    if not api_key:
        raise ValueError("API key is required: set OLLAMA_API_KEY or pass api_key")

    if os.path.exists(image_path) is False:
        raise ValueError("Image path is required: set image_path")

    print('image_path:', image_path)

    start_time = time.perf_counter()

    messages=[{
    'role': 'user',
    'content': """
        Extract all text from this image with OCR. 
        For each detected text, provide:
        1. The extracted text
        2. Confidence score (0-1 or 0-100%)
        3. Bounding box coordinates if available
        
        Format the response as JSON with this structure:
        {
            "text": "detected text",
            "confidence": 0.95,
            "bbox": [x, y, width, height]
        }""",
    'images': [image_path]}]

    response: ChatResponse = chat(
        model=model_id,
        messages=messages,
        options={'temperature': 0.0},
    )
    
    elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
    
    result = response.message.content
    print('OLLAMA_HOST=', OLLAMA_HOST)
    print('response=', response)
    print('result :', result)

    # JSON 파싱 시도
    try:
        parsed_result = json.loads(result)
        if isinstance(parsed_result, dict):
            # 필요한 필드 확인 및 기본값 설정
            return {
                "text": parsed_result.get("text", ""),
                "confidence": float(parsed_result.get("confidence", 0.0)),
                "processing_ms": parsed_result.get("duration_ms", elapsed_ms),
                "bbox": parsed_result.get("bbox", None)
            }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {e}, returning raw text")
    
    # JSON 파싱 실패 시 텍스트만 반환
    return {
        "text": result.strip(),
        "confidence": 0.0,
        "processing_ms": elapsed_ms,
        "bbox": None
    }

# def cloud_chat(
#     message: str,
#     model_id: str = "qwen3-vl:235b-instruct-cloud",
#     api_key: str | None = None,
#     stream: bool = False,
# ) -> str:
#     if api_key is None:
#         api_key = os.environ.get("OLLAMA_API_KEY")
#     if not api_key:
#         raise ValueError("API key is required: set OLLAMA_API_KEY or pass api_key")

#     # base64 encode (ascii 문자열)
#     # b64 = base64.b64encode(image_bytes).decode("ascii")
    
#     messages=[{
#     'role': 'user',
#     'content': message,
#     }]
    
#     response: ChatResponse = chat(
#         model=model_id,
#         messages=messages,
#         options={'temperature': 0.0},
#     )
    
#     return response

# def chat_stream(
#     prompt: str,
#     model: str = "qwen3-vl:235b-instruct-cloud",
#     api_key: str | None = None,
#     host: str = "https://ollama.com/api/generate",
#     timeout: int = 60,
# ):
#     """Stream the response from the generate endpoint (best-effort).

#     This will yield decoded lines as they arrive. The exact streaming format depends on
#     the server (newline-separated JSON parts is common)."""
#     if api_key is None:
#         api_key = os.environ.get("OLLAMA_API_KEY")
#     if not api_key:
#         raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {"model": model, "prompt": prompt, "stream": True}

#     with requests.post(host, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
#         resp.raise_for_status()
#         for raw in resp.iter_lines(decode_unicode=True):
#             if not raw:
#                 continue
#             # try to parse JSON chunk
#             try:
#                 part = json.loads(raw)
#             except Exception:
#                 yield raw
#                 continue
#             # extract content if present
#             text = extract_message_content(part)
#             if text:
#                 yield text
#             else:
#                 yield json.dumps(part, ensure_ascii=False)

# def _probe_models(host: str, api_key: str | None = None) -> list[str]:
#     """Try to query the /models endpoint on a set of candidate base paths and return model ids found.

#     This is a best-effort helper used for host/model validation.
#     """
#     if api_key is None:
#         api_key = os.environ.get("OLLAMA_API_KEY")
#     headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

#     # derive candidate bases from the provided host URL
#     parsed = urlparse(host)
#     base = f"{parsed.scheme}://{parsed.netloc}"
#     candidates = [f"{base}/models", f"{base}/api/models", f"{base}/models/list"]
#     results: list[str] = []
#     for url in candidates:
#         try:
#             r = requests.get(url, headers=headers, timeout=5)
#             if r.status_code == 200:
#                 data = r.json()
#                 # expect a list or dict with 'models' etc
#                 if isinstance(data, list):
#                     results.extend([str(x) for x in data])
#                 elif isinstance(data, dict):
#                     # common shapes: {'models': [...]} or {'items': [...]} or {'model': ...}
#                     for k in ("models", "items", "model"):
#                         if k in data and isinstance(data[k], list):
#                             results.extend([str(x) for x in data[k]])
#                     # sometimes API returns dict of model->meta
#                     results.extend([str(k) for k in data.keys()])
#                 if results:
#                     return results
#         except Exception:
#             continue
#     return results

# def extract_message_content(resp_json: dict[str, Any]) -> str | None:
#     if not isinstance(resp_json, dict):
#         return None
#     msg = resp_json.get("message")
#     if isinstance(msg, dict):
#         cont = msg.get("content")
#         if isinstance(cont, str):
#             return cont

#     for key in ("text", "output", "result", "completion", "body"):
#         v = resp_json.get(key)
#         if isinstance(v, str):
#             return v

#     choices = resp_json.get("choices")
#     if isinstance(choices, list) and choices:
#         c0 = choices[0]
#         if isinstance(c0, dict):
#             for k in ("text", "message", "output"):
#                 vv = c0.get(k)
#                 if isinstance(vv, str):
#                     return vv
#     return None

# def generate(
#     prompt: str,
#     model: str = "qwen3-vl:235b-instruct-cloud",
#     api_key: str | None = None,
#     host: str = "https://ollama.com/api/generate",
#     timeout: int = 30,
# ) -> str:
#     """Send a single generate request to Ollama and return the extracted text.

#     Raises requests.HTTPError on non-2xx responses.
#     """
#     if api_key is None:
#         api_key = os.environ.get("OLLAMA_API_KEY")
#     if not api_key:
#         raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {"model": model, "prompt": prompt, "stream": False}

#     resp = requests.post(host, json=payload, headers=headers, timeout=timeout)
#     # raise for HTTP errors (4xx/5xx)
#     resp.raise_for_status()

#     # attempt to parse JSON and extract content
#     try:
#         data = resp.json()
#     except ValueError:
#         # not JSON (unexpected) — return raw text
#         return resp.text.strip()

#     text = extract_message_content(data)
#     if text is None:
#         # return pretty-printed JSON as fallback
#         return json.dumps(data, ensure_ascii=False, indent=2)
#     return text

# def generate_stream(
#     prompt: str,
#     model: str = "qwen3-vl:235b-instruct-cloud",
#     api_key: str | None = None,
#     host: str = "https://ollama.com/api/generate",
#     timeout: int = 60,
# ):
#     """Stream the response from the generate endpoint (best-effort).

#     This will yield decoded lines as they arrive. The exact streaming format depends on
#     the server (newline-separated JSON parts is common)."""
#     if api_key is None:
#         api_key = os.environ.get("OLLAMA_API_KEY")
#     if not api_key:
#         raise ValueError("API key is required: set OLLAMA_API_KEY or pass --api-key")

#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {"model": model, "prompt": prompt, "stream": True}

#     with requests.post(host, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
#         resp.raise_for_status()
#         for raw in resp.iter_lines(decode_unicode=True):
#             if not raw:
#                 continue
#             # try to parse JSON chunk
#             try:
#                 part = json.loads(raw)
#             except Exception:
#                 yield raw
#                 continue
#             # extract content if present
#             text = extract_message_content(part)
#             if text:
#                 yield text
#             else:
#                 yield json.dumps(part, ensure_ascii=False)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test OCR with Ollama endpoint.")
    parser.add_argument("--image", "-i", required=True, help="Path to image file for OCR")
    parser.add_argument("--model", "-m", default="qwen3-vl:235b-instruct-cloud", help="Model id")
    parser.add_argument("--api-key", "-k", help="API key. If omitted, reads OLLAMA_API_KEY env var.")
    parser.add_argument(
        "--host",
        "-H",
        default=os.environ.get("OLLAMA_HOST", "https://ollama.com/api"),
        help="Ollama API host (default from OLLAMA_HOST or https://ollama.com/api)",
    )
    args = parser.parse_args(argv)
    
    try:
        result = ocr_image(
            image_path=args.image,
            host=args.host,
            model_id=args.model,
            api_key=args.api_key or os.environ.get("OLLAMA_API_KEY")
        )
        print("\n=== OCR Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
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
