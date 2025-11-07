import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys
NULL_OUT = open(os.devnull, "w")
STD_OUT = sys.stdout
sys.stdout = NULL_OUT

cpu_only = os.getenv('CPU_ONLY', '1') == '1'
backend = 'keras'
argv = None
exec = 'gov24.exe'
base_dir = os.path.abspath(os.path.dirname(__file__))
meipass = getattr(sys, '_MEIPASS', None)

if meipass:
    base_dir = os.path.join(meipass, "captcha_data")
else:
    base_dir = os.path.join(base_dir, "captcha_data")

from captchaResolver import dataclass, engine
from captchaResolver.keras_core import KerasModel

# def execute(captcha_type: str, image_path: str) -> str:

#     train_data = dataclass.TrainData(
#         captcha_id=captcha_type,
#         train_data_base_dir=base_dir,
#         init=False,
#         label_length=6,
#         characters=list(dataclass.DIGITS))
#     model = KerasModel(train_data=train_data, verbose=0)
#     temp_dir = os.path.abspath(base_dir)
#     temp_image_path = os.path.join(temp_dir, f"{time.time()}.png")
#     temp_image_path = os.path.abspath(temp_image_path)
    
#     with Image.open(image_path) as image:
#         if image.mode in ('RGBA', 'LA'):
#             background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
#             background.paste(image, image.split()[-1]) # omit transparency
#             image = background
#         model.train_data.image_width, model.train_data.image_height = image.size
#         image.save(temp_image_path)

#     if meipass:
#         model_path = os.path.join(base_dir, 'weights.keras')
#     else:
#         model_path = train_data.get_model_path(keras_native=model.keras_native)

#     pred, confidence = keras_engine.predict(model,temp_image_path, model_path=model_path)

#     if os.path.exists(temp_image_path):
#         os.remove(temp_image_path)

#     return pred

def predict(image_path: str) -> str:
    train_data: dataclass.TrainData = dataclass.TrainData(
        init=False,
        captcha_id="gov24",
        backend=backend,
        rev=1,
        train_data_base_dir=base_dir,
        image_width=200,
        image_height=50,
        threshold=60
    )
    captcha_type : dataclass.CaptchaType = dataclass.CaptchaType(
        name="정부 24",
        desc="대한민국 정부 24 캡챠",
        cpu_only=cpu_only,
        train_data=train_data)
    model: KerasModel = KerasModel(captcha_type=captcha_type, verbose=0)
    pred, confidence = engine.predict(model=model, image_path=image_path)
    return pred

if("__main__" == __name__):
    argv = sys.argv
    exec = os.path.basename(argv[0])

    if len(argv) < 2:
        sys.stdout = STD_OUT
        print('사용법 : ' + exec + ' <이미지파일경로>')
        print('해당 캡챠 이미지를 인식한 결과를 반환합니다.')
        print('<이미지파일경로>는 인식할 이미지 파일의 경로를 입력합니다. 예) "C:\\temp\\download.png"')
        sys.exit(-1)

    image_path = argv[1]
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        sys.stdout = STD_OUT
        print(f'에러 : 이미지 파일을 찾을 수 없습니다. {image_path}')
        sys.exit(-1)

    # pred = execute('gov24', image_path=image_path)
    pred = predict(image_path=image_path)
    sys.stdout = STD_OUT
    sys.stdout.write(pred)
    sys.exit(0)

sys.exit(0)
