import pytest

from web.services import captcha as captcha_service


@pytest.fixture(autouse=True)
def clear_model_cache():
	captcha_service._MODEL_CACHE.clear()
	yield
	captcha_service._MODEL_CACHE.clear()


@pytest.fixture
def broken_wetax(tmp_path, monkeypatch, captcha_data_dir):
	"""웹 서비스가 테스트 전용 데이터 디렉터리를 보도록 한다."""
	captcha_data_dir("wetax", labels=("012345", "678901", "234567"), size=(200, 60))
	monkeypatch.setattr(captcha_service, "CAPTCHA_DATA_DIR", tmp_path / "captcha_data")


def test_get_model_raises_when_weights_cannot_be_loaded(broken_wetax):
	with pytest.raises(Exception):
		captcha_service.get_model("wetax", "cpu")


def test_get_model_caches_nothing_when_weights_cannot_be_loaded(broken_wetax):
	with pytest.raises(Exception):
		captcha_service.get_model("wetax", "cpu")

	assert captcha_service._MODEL_CACHE == {}


def test_second_request_still_fails_instead_of_serving_untrained_model(broken_wetax):
	"""첫 요청만 500 이고 두 번째부터 무학습 모델로 답하던 버그."""
	with pytest.raises(Exception):
		captcha_service.get_model("wetax", "cpu")

	with pytest.raises(Exception):
		captcha_service.get_model("wetax", "cpu")


def test_failed_load_is_not_reported_as_loaded(broken_wetax):
	with pytest.raises(Exception):
		captcha_service.get_model("wetax", "cpu")

	assert captcha_service.loaded_model_ids() == []
