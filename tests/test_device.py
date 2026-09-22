import sys
from unittest.mock import MagicMock

import pytest

from web.core import device


@pytest.fixture
def torch_stub(monkeypatch):
	torch = MagicMock()
	torch.__version__ = "test"
	torch.version.cuda = "13.0"
	torch.cuda.is_available.return_value = True
	torch.cuda.device_count.return_value = 1
	torch.cuda.get_device_name.return_value = "Test GPU"
	monkeypatch.setitem(sys.modules, "torch", torch)
	# 실행 검사 캐시는 프로세스마다 유지하며 테스트 사이에는 초기화한다.
	device._cuda_execution_error.cache_clear()
	yield torch
	device._cuda_execution_error.cache_clear()


def test_auto_uses_cpu_when_gpu_is_detected_but_kernels_cannot_run(torch_stub):
	torch_stub.ones.side_effect = RuntimeError("no kernel image is available")
	assert device.resolve("auto") == "cpu"
	assert device.resolve(None) == "cpu"
	options = device.device_options()
	assert options[0]["label"] == "자동 (cpu)"
	assert options[2]["available"] is False
	assert "no kernel image" in options[2]["detail"]


def test_explicit_cuda_reports_execution_failure(torch_stub):
	torch_stub.ones.side_effect = RuntimeError("no kernel image is available")
	with pytest.raises(ValueError, match="CUDA 를 사용할 수 없습니다"):
		device.resolve("cuda")


def test_working_cuda_is_checked_once(torch_stub):
	assert device.resolve("auto") == "cuda"
	assert device.resolve("cuda") == "cuda"
	assert device.cuda_status()["available"] is True
	torch_stub.ones.assert_called_once_with(1, device="cuda")
	torch_stub.ones.return_value.add_.return_value.item.assert_called_once()


def test_explicit_cpu_does_not_probe_cuda(torch_stub):
	assert device.resolve("cpu") == "cpu"
	torch_stub.ones.assert_not_called()


@pytest.mark.parametrize("built", [None, "13.0"])
def test_no_gpu_uses_cpu_without_probe(torch_stub, built):
	torch_stub.version.cuda = built
	torch_stub.cuda.is_available.return_value = False
	assert device.resolve("auto") == "cpu"
	assert device.cuda_status()["available"] is False
	torch_stub.ones.assert_not_called()


def test_invalid_device_is_rejected(torch_stub):
	with pytest.raises(ValueError, match="unsupported device"):
		device.resolve("invalid")
