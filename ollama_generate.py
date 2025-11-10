from __future__ import annotations

import os
import json
import time
from ollama import Client, ChatResponse

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'https://ollama.com')
os.environ.setdefault('OLLAMA_HOST', 'https://ollama.com')
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
    
    from ollama import Client

    client = Client(
        host=OLLAMA_HOST,
        headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
    )

    response: ChatResponse = client.chat(
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

def chat_text(
    message: str,
    stream: bool = False,
    model_id: str = OLLAMA_MODEL_ID,
    api_key: str = OLLAMA_API_KEY,
):
    start_time = time.perf_counter()
    
    # 스트림 모드에 따라 다른 시스템 메시지 사용
    if stream:
        system_msg = """You are a helpful assistant.
Answer questions directly and concisely in Korean.
IMPORTANT: Return ONLY plain text in your response. 
Do NOT use JSON format, markdown formatting, or any structured format.
Just provide the direct answer as natural text."""
    else:
        system_msg = """You are a helpful assistant that answers in Korean.
Format your response as JSON with this exact structure:
{
    "text": "your answer here",
    "confidence": 0.95,
    "processing_ms": 1234
}"""
    user_msg: dict = {'role': 'user', 'content': message}
    client = Client(
        host=OLLAMA_HOST,
        headers={'Authorization': 'Bearer ' + (api_key or OLLAMA_API_KEY)}
    )

    # Streaming mode: return a generator that yields chunks as they arrive
    if stream:
        def _stream_generator():
            try:
                stream_resp = client.chat(
                    model=model_id,
                    messages=[{'role': 'system', 'content': system_msg}, user_msg],
                    options={'temperature': 0.0, 'max_tokens': 1024 * 4},
                    stream=True,
                )
            except Exception as e:
                elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
                yield json.dumps({"error": str(e), "processing_ms": elapsed_ms}, ensure_ascii=False)
                return

            for part in stream_resp:
                text = None
                try:
                    if hasattr(part, "message"):
                        text = getattr(part.message, "content", None)
                    elif isinstance(part, dict):
                        # Extract text field from dict, don't serialize the whole dict
                        text = part.get("text") or part.get("content") or part.get("response") or part.get("answer")
                        # if nested message shape
                        if text is None and part.get("message"):
                            msg = part.get("message")
                            if isinstance(msg, dict):
                                text = msg.get("content") or msg.get("text")
                            elif isinstance(msg, str):
                                text = msg
                    elif isinstance(part, str):
                        text = part
                except Exception:
                    text = None

                # If we still don't have text and part is dict-like, try to extract any textual content
                if text is None and isinstance(part, dict):
                    # Last resort: try to find any string value that looks like content
                    for key in ['text', 'content', 'response', 'answer', 'output']:
                        if key in part and isinstance(part[key], str):
                            text = part[key]
                            break

                # Only convert to string if we have something meaningful
                if text is not None:
                    if not isinstance(text, str):
                        try:
                            text = str(text)
                        except Exception:
                            text = ""
                    
                    text = text.strip()
                    if text:
                        yield text

        return _stream_generator()

    # Non-streaming (synchronous) behavior
    response = None
    last_exc = None
    for attempt in range(2):
        try:
            response: ChatResponse = client.chat(
                model=model_id,
                messages=[{'role': 'system', 'content': system_msg}, user_msg],
                options={'temperature': 0.0, 'max_tokens': 1024 * 4},
            )
            break
        except Exception as e:
            last_exc = e
            if attempt == 0:
                time.sleep(0.25)
            else:
                elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
                import traceback
                tb = traceback.format_exc()
                print(f"chat_text failed: {e}\n{tb}")
                return {"text": f"<error: {str(e)}>", "confidence": 0.0, "processing_ms": elapsed_ms}

    elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
    result = getattr(response.message, 'content', '') if response is not None else ''

    if not isinstance(result, str):
        try:
            result = str(result)
        except Exception:
            result = ''

    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            resp_text = parsed.get('text') or parsed.get('response') or parsed.get('answer') or result
            confidence = float(parsed.get('confidence', 0.0))
            processing_ms = int(parsed.get('processing_ms', elapsed_ms))
            return {"text": resp_text, "confidence": confidence, "processing_ms": processing_ms}
    except Exception:
        pass

    return {"text": result.strip(), "confidence": 0.0, "processing_ms": elapsed_ms}
