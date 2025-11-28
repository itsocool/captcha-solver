from calendar import c
import os
import shutil
import numpy as np
import keras
import tensorflow as tf
from typing import Optional, Tuple, List, Union
from keras import layers, models, ops, saving
from captchaResolver.dataclass import CaptchaType, TrainData
from captchaResolver.base_core import BaseModel

@saving.register_keras_serializable(package="captchaResolver")
class CTCLayer(layers.Layer):
    def __init__(
        self,
        from_logits: bool = False,
        blank_index: int = -1,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.blank_index = blank_index
        self.supports_masking = True  # CTC handles variable-length sequences

    def get_config(self) -> dict:
        """Return layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration including:
            - from_logits: Logits mode setting
            - blank_index: Blank token index
            - Plus all base layer config (name, dtype, etc.)
        """
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
            "blank_index": self.blank_index,
        })
        return config

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        batch_size = ops.shape(y_true)[0]
        time_steps = ops.shape(y_pred)[1]
        max_label_len = ops.shape(y_true)[1]
        input_length = ops.cast(
            ops.repeat(time_steps, batch_size),
            dtype=tf.int32
        )
        input_length = ops.reshape(input_length, (batch_size, 1))
        label_length = ops.cast(
            ops.repeat(max_label_len, batch_size),
            dtype=tf.int32
        )
        label_length = ops.reshape(label_length, (batch_size, 1))
        loss = self.ctc_batch_cost(
            y_true=y_true,
            y_pred=y_pred,
            input_length=input_length,
            label_length=label_length,
            from_logits=self.from_logits,
            blank_index=self.blank_index
        )
        
        self.add_loss(loss)
        return y_pred

    def ctc_label_dense_to_sparse(
        self,
        labels: tf.Tensor, 
        label_lengths: tf.Tensor
    ) -> tf.SparseTensor:
        label_shape = ops.shape(labels)
        batch_size = label_shape[0]
        max_label_len = label_shape[1]
        label_positions = ops.arange(max_label_len)  # [0, 1, 2, ..., max_label_len-1]
        label_lengths_expanded = ops.expand_dims(label_lengths, axis=1)  # (batch_size, 1)
        valid_positions_mask = label_positions < label_lengths_expanded  # (batch_size, max_label_len)
        valid_indices = tf.where(valid_positions_mask)
        valid_label_values = tf.gather_nd(labels, valid_indices)
        return tf.SparseTensor(
            indices=ops.cast(valid_indices, dtype=tf.int64),
            values=valid_label_values,
            dense_shape=ops.cast(label_shape, dtype=tf.int64)
        )

    def _normalize_ctc_inputs(
        self,
        input_length: tf.Tensor,
        label_length: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype=tf.int32)
        input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype=tf.int32)
        return input_length, label_length

    def _convert_to_time_major_logits(
        self,
        y_pred: tf.Tensor,
        from_logits: bool = False
    ) -> tf.Tensor:
        y_pred_time_major = ops.transpose(y_pred, axes=[1, 0, 2])
        if from_logits:
            return y_pred_time_major
        else:
            epsilon = keras.config.epsilon()
            return ops.log(y_pred_time_major + epsilon)

    def ctc_batch_cost(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        input_length: tf.Tensor,
        label_length: tf.Tensor,
        from_logits: bool = False,
        blank_index: int = -1
    ) -> tf.Tensor:
        input_length, label_length = self._normalize_ctc_inputs(input_length, label_length)
        sparse_labels = ops.cast(
            self.ctc_label_dense_to_sparse(y_true, label_length),
            dtype=tf.int32
        )
        logits = self._convert_to_time_major_logits(y_pred, from_logits=from_logits)
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=logits,
            label_length=None,  # Already encoded in sparse_labels structure
            logit_length=input_length,
            logits_time_major=True,  # We transposed to time-major format
            blank_index=blank_index,
        )
        return ops.expand_dims(loss, axis=1)

class KerasModel(BaseModel):
    """
    Keras/TensorFlow 기반 CAPTCHA 인식 모델
    
    BaseModel을 상속받아 구현합니다.
    """

    def __init__(self, captcha_type: CaptchaType, verbose=1, keras_model=True, saved_model=False):
        super().__init__(captcha_type, verbose)
        self.char_to_num = layers.StringLookup(
            vocabulary=self.train_data.characters, mask_token=None, num_oov_indices=0
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.predict_model = None
        self.keras_model = keras_model
        self.saved_model = saved_model

    def split_dataset(self, batch_size=32, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        images = np.array(self.train_data.get_data_files(train=True))
        labels = np.array(self.train_data.get_labels(train=True))
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = (
            images[indices[:train_samples]],
            labels[indices[:train_samples]],
        )
        x_valid, y_valid = (
            images[indices[train_samples:]],
            labels[indices[train_samples:]],
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset
            .map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .cache()  # Cache validation set
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return train_dataset, validation_dataset

    def encode_single_sample(self, image_path, label = None):
        image_width = self.train_data.image_width
        image_height = self.train_data.image_height
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        threshold = self.train_data.threshold

        if threshold > 0:
            image = tf.where(image > threshold, 255.0, image)

        image = tf.image.resize(image, [image_height, image_width])
        image = tf.transpose(image, perm=[1, 0, 2])

        if label is not None:
            label = self.char_to_num(
                tf.strings.unicode_split(label, input_encoding="UTF-8")
            )

        return {"image": image, "label": label}

    def build_model(self, prediction_only=False) -> models.Model:
        # Inputs to the model
        width, height = self.train_data.image_width, self.train_data.image_height
        # 공통 feature extractor
        input_img = layers.Input(shape=(width, height, 1), name="image", dtype="float32")

        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        new_shape = (
            (self.train_data.image_width // 4),
            (self.train_data.image_height // 4) * 64,
        )
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer (softmax)
        unit = len(list(self.train_data.characters)) + 1
        x = layers.Dense(unit, activation="softmax", name="dense2")(x)

        if prediction_only:
            # 추론 전용 모델: 이미지 입력 -> softmax 출력
            pred_model = keras.models.Model(inputs=input_img, outputs=x, name="ocr_prediction_v1")
            # 예측용 모델은 컴파일 불필요
            return pred_model
        else:
            # 학습용 모델: labels 입력 및 CTCLayer 포함
            labels = layers.Input(name="label", shape=(None,), dtype="float32")
            output = CTCLayer(name="ctc_loss")(labels, x)
            model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
            optimizer = keras.optimizers.Adam()
            model.compile(optimizer=optimizer)
            return model

    def ctc_decode(
        self,
        y_pred: tf.Tensor,
        input_length: Union[tf.Tensor, np.ndarray],
        greedy: bool = False,
        beam_width: int = 100,
        top_paths: int = 1
    ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        input_shape = ops.shape(y_pred)
        batch_size, time_steps = input_shape[0], input_shape[1]
        y_pred_log = ops.log(
            ops.transpose(y_pred, axes=[1, 0, 2]) + keras.config.epsilon()
        )
        input_length = ops.cast(input_length, dtype=tf.int32)
        if greedy:
            decoded, log_probabilities = tf.nn.ctc_greedy_decoder(
                inputs=y_pred_log,
                sequence_length=input_length,
                merge_repeated=True  # Collapse repeated characters (CTC behavior)
            )
        else:
            decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
                inputs=y_pred_log,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=top_paths,
            )
        
        decoded_dense = []
        for sparse_tensor in decoded:
            reconstructed = tf.SparseTensor(
                indices=sparse_tensor.indices,
                values=sparse_tensor.values,
                dense_shape=(batch_size, time_steps)
            )
            dense_tensor = tf.sparse.to_dense(
                sp_input=reconstructed,
                default_value=-1
            )
            decoded_dense.append(dense_tensor)
        
        return decoded_dense, log_probabilities

    def decode_batch_predictions(self, pred, use_greedy=False):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Decode using CTC decoder (greedy or beam search)
        results = self.ctc_decode(
            pred, 
            input_length=input_len, 
            greedy=use_greedy,
            beam_width=100,
            top_paths=1
        )[0][0][:, : self.train_data.label_length]
        
        # Convert indices to text
        output_text = []
        for res in results:
            # Add 1 to shift indices (CTC uses 0 for blank)
            res = (
                tf.strings.reduce_join(self.num_to_char(res + 1))
                .numpy()
                .decode("utf-8")
            )
            output_text.append(res)
        return output_text

    def load_prediction_model(self, model_path: str = None) -> models.Model:
        if model_path is None:
            model_path = self.train_data.get_model_path()

        if model_path is None:
            raise ValueError("model_path resolved to None. Ensure TrainData.get_model_path() returns a valid path")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Check that the model exists in {self.train_data.get_model_base_dir()}"
            )

        custom_objects = {"CTCLayer": CTCLayer}
        loaded = keras.models.load_model(model_path, custom_objects=custom_objects)
        input_layer = loaded.input[0] if isinstance(loaded.input, list) else loaded.input
        output_layer = loaded.get_layer(name="dense2").output
        self.predict_model = keras.models.Model(input_layer, output_layer)
        return self.predict_model

    def train_model(
        self,
        epochs=100,
        batch_size=32,
        earlystopping=True,
        early_stopping_patience: int = 8,
    ) -> models.Model:
        train_dataset, validation_dataset = self.split_dataset(
            batch_size=batch_size, train_size=0.9, shuffle=True
        )
        train_data = self.train_data
        train_model = self.build_model()
        model_base_dir = train_data.get_model_base_dir()
        os.makedirs(model_base_dir, exist_ok=True)
        best_model_path = os.path.join(model_base_dir, "best_weights.keras")
        callbacks = []
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=1 if self.verbose else 0,
            )
        )
        if earlystopping:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=False,  # We'll load from checkpoint instead
                verbose=0,  # Disable default verbose, use custom callback
            )
            callbacks.append(early_stopping_callback)
            class PatienceDisplayCallback(keras.callbacks.Callback):
                def __init__(self, early_stopping_cb, verbose=True):
                    super().__init__()
                    self.early_stopping_cb = early_stopping_cb
                    self.verbose = verbose
                    self.best_loss = float('inf')
                
                def on_epoch_end(self, epoch, logs=None):
                    if not self.verbose:
                        return
                    
                    current_loss = logs.get('val_loss')
                    if current_loss is None:
                        return
                    
                    # Check if loss improved
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        print(f"  ✓ 검증 손실 개선: {current_loss:.6f} (최고 기록 갱신)")
                    else:
                        wait = self.early_stopping_cb.wait
                        remaining = early_stopping_patience - wait
                        print(f"  ⚠ 검증 손실 미개선 - 남은 patience: {remaining}/{early_stopping_patience}")
                        
                        if wait >= early_stopping_patience - 1:
                            print(f"  ⚠ 조기 종료 예정 (다음 에포크에서 개선 없으면 종료)")
            
            callbacks.append(PatienceDisplayCallback(early_stopping_callback, verbose=self.verbose))

        # Train model
        train_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        # Save best_weights -> final weights.keras (복사 또는 현재 모델 저장)
        final_model_path = os.path.join(model_base_dir, "weights.keras")
        try:
            if os.path.exists(best_model_path):
                # best가 있으면 복사해서 final로 만듦
                shutil.copy2(best_model_path, final_model_path)
                # 복사 성공 시 임시 파일 제거
                try:
                    os.remove(best_model_path)
                except OSError:
                    # 삭제 실패해도 진행
                    pass
                if self.verbose:
                    print(f"\n✓ 학습 완료:")
                    print(f"  - 최종 모델: {final_model_path}")
                    print(f"  - 임시 best 파일(best_weights.keras)을 final로 복사하고 정리함.")
            else:
                # best가 없으면 현재(학습중) 모델을 저장
                train_model.save(final_model_path)
                if self.verbose:
                    print(f"\n✓ 학습 완료: best model이 없어 현재 모델을 저장합니다.")
                    print(f"  - 최종 모델: {final_model_path}")
        except Exception as e:
            # 복사/저장 중 에러가 나면 fallback으로 현재 모델 저장 시도
            try:
                train_model.save(final_model_path)
                if self.verbose:
                    print(f"\n✓ 학습 완료: 복사 실패했으나 현재 모델을 저장했습니다. ({e})")
                    print(f"  - 최종 모델: {final_model_path}")
            except Exception as e2:
                if self.verbose:
                    print(f"\n⚠ 모델 저장 실패: {e2}")
        
        return train_model

    def batch_predict(
        self, batch_size=32, use_greedy=False
    ):
        model = self
        pred_img_path_list = model.train_data.get_data_files(train=False)
        pred_labels = model.train_data.get_labels(train=False)
        pred_dataset = tf.data.Dataset.from_tensor_slices((pred_img_path_list, pred_labels))
        pred_dataset = (
            pred_dataset
            .map(model.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        model.load_prediction_model()
        all_preds = []
        all_labels = []
        all_confidences = []
        
        for batch in pred_dataset:
            images = batch["image"]
            labels = batch["label"]
            pred_vals = model.predict_model.predict(images, batch_size=batch_size, verbose=0)
            preds = model.decode_batch_predictions(pred_vals, use_greedy=use_greedy)
            for label in labels:
                label_text = tf.strings.reduce_join(
                    model.num_to_char(label + 1)
                ).numpy().decode("utf-8")
                all_labels.append(label_text)
            
            all_preds.extend(preds)
            batch_confidences = np.mean(np.max(pred_vals, axis=2), axis=1) * 100.0
            all_confidences.extend([float(c) for c in batch_confidences.tolist()])
        return all_preds, all_labels, all_confidences, pred_img_path_list

    def predict(
        self,
        image_path: str,
        use_greedy: bool = False,
    ) -> Tuple[str, float]:
        self.load_prediction_model()
        sample = self.encode_single_sample(image_path)
        image = ops.expand_dims(sample["image"], axis=0)
        pred = self.predict_model.predict(image)
        decoded_preds = self.decode_batch_predictions(pred, use_greedy=use_greedy)
        confidence = np.max(pred)
        return decoded_preds[0], confidence
