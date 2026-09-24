"""브라우저 중단 시 수집 락 반환과 SSE 오류 전달 회귀."""
import asyncio
import threading

import pytest
from starlette.requests import ClientDisconnect
from web.api.v1 import data_source as api
from web.services import data_source as service


@pytest.mark.parametrize('kind', ['collect', 'auto_label', 'confidence'])
def test_disconnected_response_closes_worker_and_releases_lock(monkeypatch, kind):
    lock = threading.Lock()
    monkeypatch.setattr(service, '_RUN_LOCK', lock)
    monkeypatch.setattr(api, 'clean_request', lambda *args: {})
    def events(*args, **kwargs):
        assert lock.acquire(blocking=False)
        try:
            yield {'type': 'start'}
            yield {'type': 'item'}
        finally:
            lock.release()
    monkeypatch.setattr(api, 'run', events)
    monkeypatch.setattr(api, 'iter_auto_label', events)
    async def scenario():
        if kind == 'collect':
            response = await api.data_source_stream('iros', 1, 'https://example.test', '', 1, 0, 'image')
        elif kind == 'auto_label':
            response = await api.data_source_auto_label_stream('iros', 1, None, 0.)
        else:
            response = await api.data_source_confidence_stream('iros', 1, None)
        async def send(message):
            if message['type'] == 'http.response.body':
                assert lock.locked()
                raise OSError('브라우저 연결 종료')
        async def receive():
            return {'type': 'http.disconnect'}
        with pytest.raises(ClientDisconnect):
            await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
        # response가 GC되기 전에 반드시 해제되어 다음 추가가 가능해야 한다.
        assert not lock.locked()
    asyncio.run(scenario())


def test_asgi_disconnect_signal_closes_after_inflight_work(monkeypatch):
    lock = threading.Lock()
    monkeypatch.setattr(service, '_RUN_LOCK', lock)
    monkeypatch.setattr(api, 'clean_request', lambda *args: {})
    def events(*args):
        with lock:
            yield {'type': 'start'}
            yield {'type': 'item'}
    monkeypatch.setattr(api, 'run', events)
    async def scenario():
        response = await api.data_source_stream('iros', 1, 'https://example.test', '', 1, 0, 'image')
        emitted = asyncio.Event()
        async def send(message):
            if message['type'] == 'http.response.body':
                emitted.set()
                await asyncio.Event().wait()
        async def receive():
            await emitted.wait()
            return {'type': 'http.disconnect'}
        await asyncio.wait_for(response({'type': 'http', 'asgi': {'spec_version': '2.3'}}, receive, send), 2)
        assert not lock.locked()
        # 두 번째 요청이 즉시 시작·완료될 수 있다.
        assert [event['type'] for event in events()] == ['start', 'item']
    asyncio.run(scenario())


def test_collection_saves_images_and_accepts_next_request(tmp_path, monkeypatch):
    import io
    from PIL import Image
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.core import engine

    lock = threading.Lock()
    monkeypatch.setattr(service, '_RUN_LOCK', lock)
    monkeypatch.setattr(service, 'CAPTCHA_DATA_DIR', tmp_path)
    monkeypatch.setattr(service, '_persist_params', lambda *args: None)
    monkeypatch.setattr(engine, 'get_captcha_type_list', lambda **kwargs: {'iros': object()})
    content = io.BytesIO()
    Image.new('L', (120, 40), 255).save(content, format='PNG')
    monkeypatch.setattr(service, '_fetch', lambda *args: (content.getvalue(), 'image/png'))
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        for count in (1, 2):
            response = client.get('/data-source/stream', params=dict(
                captcha_id='iros', rev=1, url='https://example.test/image.png', count=1))
            assert response.status_code == 200
            assert 'event: summary' in response.text and '"saved": 1' in response.text
            assert not lock.locked()
            assert len(list(service.draft_dir('iros', 1).glob('*.png'))) == count
