import time
import keras
import numpy as np
import tensorflow as tf

from PIL import Image
from captchaResolver.core import KerasModel
from captchaResolver.dataclass import CaptchaType

def train_model(
    model: KerasModel,
    epochs=100,
    batch_size=32,
    hard_mode=False,
    earlystopping=True,
    early_stopping_patience: int = 8
):
    train_dataset, validation_dataset = model.split_dataset(
        batch_size=batch_size, train_size=0.9, shuffle=True
    )
    train_data = model.train_data
    train_model = model.build_model()
    callbacks = []
    
    if earlystopping == True:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            )
        )
        train_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=model.verbose,
        )
    else:
        train_model.fit(
            train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=model.verbose,
        )
    model_path = train_data.get_model_path(keras_native=model.keras_native)
    print("model_path : ", model_path)

    if model.keras_native:
        train_model.save(model_path)
    else:
        tf.saved_model.save(train_model, model_path)

def batch_predict_model(model: KerasModel, batch_size=32):
    start = time.time()
    matched = 0
    pred_img_path_list = model.train_data.get_data_files(train=False)
    pred_labels = model.train_data.get_labels(train=False)
    pred_dataset = tf.data.Dataset.from_tensor_slices((pred_img_path_list, pred_labels))
    pred_dataset = (
        pred_dataset
        .map(model.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    # Load prediction model if not loaded
    model.load_prediction_model()
    
    # Batch prediction
    all_preds = []
    all_labels = []
    
    for batch in pred_dataset:
        images = batch["image"]
        labels = batch["label"]
        
        # Predict batch
        pred_vals = model.predict_model.predict(images, verbose=0)
        preds = model.decode_batch_predictions(pred_vals)
        
        # Decode original labels
        for label in labels:
            label_text = tf.strings.reduce_join(
                model.num_to_char(label + 1)
            ).numpy().decode("utf-8")
            all_labels.append(label_text)
        
        all_preds.extend(preds)
    
    # Compare predictions with original labels
    for idx, (ori, pred) in enumerate(zip(all_labels, all_preds)):
        msg = ""
        if ori == pred:
            matched += 1
        else:
            msg = " Not matched!"
        
        # Calculate confidence for display (optional)
        print(f"ori: {ori}, pred: {pred}{msg}")
    
    end = time.time()
    total = len(pred_img_path_list)
    accuracy = matched / total * 100 if total > 0 else 0
    
    print(f"Matched: {matched}, Total: {total}, Accuracy: {accuracy:.2f}%")
    print(f"pred time: {end - start:.2f} sec")

def predict(model: KerasModel, image_path: str, model_path: str) -> tuple[str, float]:
    
    image_width = model.train_data.image_width
    image_height = model.train_data.image_height
    target_img = model.encode_single_sample(image_path)["image"]
    target_img = tf.reshape(target_img, shape=[1, image_width, image_height, 1])

    if model.predict_model is None:
        model.load_prediction_model()

    pred_val = model.predict_model.predict(target_img, verbose=model.verbose)
    pred = model.decode_batch_predictions(pred_val)[0]

    confidence = float(np.max(pred_val, axis=-1).mean())

    return pred, confidence
