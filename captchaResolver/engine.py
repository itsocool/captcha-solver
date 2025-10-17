import time
import keras
import tensorflow as tf

from captchaResolver.core import KerasModel
from captchaResolver.dataclass import CaptchaType

def get_captcha_type_list(image_dir: str = "./images", model_dir: str = "./model"):
    default = CaptchaType(id="default", name="기본 캡챠", desc="기본 캡챠")
    supreme_court = CaptchaType(id="supreme_court", name="대법원", desc="대법원 캡챠")
    gov24 = CaptchaType(id="gov24", name="gov24", desc="대한민국 정부 24 캡챠")
    wetax = CaptchaType(id="wetax", name="wetax", desc="WETAX 캡챠")
    kshop = CaptchaType(id="kshop", name="kshop", desc="KT Shopping 캡챠")

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }

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

def predict_model(model: KerasModel, batch_size=32):
    """Validate model using dataset-based batch prediction.
    
    Args:
        model: KerasModel instance
        batch_size: Batch size for prediction
    """
    start = time.time()
    matched = 0
    
    # Get prediction files and labels
    train_data = model.train_data
    pred_img_path_list = model.train_data.get_data_files(train=False)
    pred_labels = model.train_data.get_labels(train=False)
    
    # Create dataset for batch processing
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
