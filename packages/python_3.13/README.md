# aso-ai

Python 3.13 이상의 PyTorch CRNN CAPTCHA 인식 라이브러리입니다.
배포 이름은 `aso-ai`, import 이름은 `aso_ai`입니다.

```bash
uv build --project packages/python_3.13
uv run --project apps/web aso-ai -c supreme_court -i path/to/image.png
```

```python
from web.core import engine

model = engine.get_captcha_model(train_data_base_dir="./captcha_data", captcha_id="supreme_court")
text, confidence = engine.predict(model=model, image_path="sample.png")
```

`import aso_ai`는 torch를 로드하지 않습니다. `aso_ai.core` 또는 `web.core.engine`을
import할 때 PyTorch/CUDA 초기화가 실행됩니다. 모델 구현은 `src/aso_ai/core.py`,
실행 엔진과 데이터 모델은 `apps/web/core/engine.py`·`dataclass.py`에 있습니다.
CLI는 웹 엔진을 사용하므로 `web` 패키지와 함께 설치합니다. `aso-ai` 명령은 웹 프로젝트가 등록합니다.

학습과 일괄 평가는 웹 `/train`·`/predict` 또는 `engine.train_model()`·
`engine.batch_predict_model()`을 사용합니다. 하드코딩된 `train.py`·`pred.py`
실행 스크립트는 제거했으며, 단건 CLI `aso-ai`와 `python -m aso_ai`는 유지합니다.
