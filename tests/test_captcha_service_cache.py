import pytest

from web.services import captcha as captcha_service


@pytest.fixture(autouse=True)
def clear_model_cache():
	captcha_service._MODEL_CACHE.clear()
	yield
	captcha_service._MODEL_CACHE.clear()


@pytest.fixture
def broken_kshop(tmp_path, monkeypatch, captcha_data_dir):
	"""engine 기본 base_dir 이 './captcha_data' 라 cwd 를 픽스처 루트로 옮긴다."""
	captcha_data_dir("kshop", labels=("012345", "678901", "234567"), size=(263, 54))
	monkeypatch.chdir(tmp_path)


def test_get_model_raises_when_weights_cannot_be_loaded(broken_kshop):
	with pytest.raises(Exception):
		captcha_service.get_model("kshop", "cpu")


def test_get_model_caches_nothing_when_weights_cannot_be_loaded(broken_kshop):
	with pytest.raises(Exception):
		captcha_service.get_model("kshop", "cpu")

	assert captcha_service._MODEL_CACHE == {}


def test_second_request_still_fails_instead_of_serving_untrained_model(broken_kshop):
	"""첫 요청만 500 이고 두 번째부터 무학습 모델로 답하던 버그."""
	with pytest.raises(Exception):
		captcha_service.get_model("kshop", "cpu")

	with pytest.raises(Exception):
		captcha_service.get_model("kshop", "cpu")


def test_failed_load_is_not_reported_as_loaded(broken_kshop):
	with pytest.raises(Exception):
		captcha_service.get_model("kshop", "cpu")

	assert captcha_service.loaded_model_ids() == []
