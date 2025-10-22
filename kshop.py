import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, time
NULL_OUT = open(os.devnull, "w")
STD_OUT = sys.stdout
sys.stdout = NULL_OUT

argv = None
exec = 'kshop.exe'
base_dir = os.path.abspath(os.path.dirname(__file__))
meipass = getattr(sys, '_MEIPASS', None)

if meipass:
    base_dir = os.path.join(meipass, "model")
else:
    base_dir = os.path.join(base_dir, "captcha_data")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from captchaResolver import dataclass, keras_engine
from PIL import Image
from captchaResolver.keras_core import KerasModel

def execute(captcha_type: str, image_path: str) -> str:

    train_data = dataclass.TrainInfo(
        captcha_id=captcha_type,
        captcha_data_base_dir=base_dir,
        init=False,
        label_length=6,
        characters=list(dataclass.DIGITS))
    model = KerasModel(train_data=train_data, verbose=0)
    temp_dir = os.path.abspath(base_dir)
    temp_image_path = os.path.join(temp_dir, f"{time.time()}.png")
    temp_image_path = os.path.abspath(temp_image_path)
    
    with Image.open(image_path) as image:
        if image.mode in ('RGBA', 'LA'):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1]) # omit transparency
            image = background
        model.train_data.image_width, model.train_data.image_height = image.size
        image.save(temp_image_path)

    if meipass:
        model_path = os.path.join(base_dir, 'weights.keras')
    else:
        model_path = train_data.get_model_path(keras_native=model.keras_native)

    pred, confidence = keras_engine.predict(model,temp_image_path, model_path=model_path)

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

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

    pred = execute('kshop', image_path=image_path)
    sys.stdout = STD_OUT
    sys.stdout.write(pred)
    sys.exit(0)

sys.exit(0)
