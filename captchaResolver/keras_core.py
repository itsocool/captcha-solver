import os
import keras
import numpy as np
import tensorflow as tf
from keras import layers, models, backend, callbacks

from captchaResolver.dataclass import TrainInfo

def ctc_label_dense_to_sparse(labels, label_lengths):
    """Convert dense labels to sparse format for CTC.
    
    Improved version using vectorized operations instead of tf.scan()
    for better memory efficiency and performance.
    """
    label_shape = tf.shape(labels)
    batch_size = label_shape[0]
    max_label_len = label_shape[1]

    # Vectorized mask generation using broadcasting
    # Shape: (batch_size, max_label_len)
    label_positions = tf.range(max_label_len)  # [0, 1, 2, ..., max_label_len-1]
    label_lengths_expanded = tf.expand_dims(label_lengths, axis=1)  # (batch_size, 1)
    
    # Broadcasting comparison: (batch_size, 1) vs (max_label_len,) -> (batch_size, max_label_len)
    dense_mask = label_positions < label_lengths_expanded

    # Use tf.where to get indices of True values
    # Returns shape (num_valid, 2) where each row is [batch_idx, label_idx]
    indices = tf.where(dense_mask)

    # Extract values at the valid positions
    vals_sparse = tf.gather_nd(labels, indices)

    return tf.SparseTensor(
        tf.cast(indices, dtype="int64"), 
        vals_sparse, 
        tf.cast(label_shape, dtype="int64")
    )

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """CTC batch cost compatible with TF/Keras 3.

    Args:
        y_true: dense integer labels tensor, shape (batch, max_label_len)
        y_pred: prediction probs (softmax) tensor, shape (batch, time, num_classes)
        input_length: tensor with shape (batch, 1) or (batch,) indicating time steps
        label_length: tensor with shape (batch, 1) or (batch,) indicating label lengths

    Returns:
        loss: tensor with shape (batch, 1)
    """
    # Ensure proper dtypes and shapes
    # label_length / input_length may come in as (batch, 1)
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), dtype="int32")
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), dtype="int32")

    # Convert dense labels to sparse required by tf.nn.ctc_loss
    sparse_labels = tf.cast(ctc_label_dense_to_sparse(y_true, label_length), dtype="int32")

    # tf.nn.ctc_loss expects time-major logits: (time, batch, num_classes)
    # y_pred is expected to be probabilities (softmax). Convert to log-probs to be numerically stable.
    y_pred_time_major = tf.transpose(y_pred, perm=[1, 0, 2])
    log_probs = tf.math.log(y_pred_time_major + backend.epsilon())

    # Use TF native ctc_loss
    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=log_probs,
        label_length=None,
        logit_length=input_length,
        logits_time_major=True,
        blank_index=-1,
    )

    # Expand dims to match Keras backend shape (batch, 1)
    return tf.expand_dims(loss, 1)

def ctc_decode(y_pred, input_length, beam_width=100, top_paths=1):
    """Decode CTC predictions to text.
    
    Args:
        y_pred: Prediction tensor
        input_length: Length of input sequences
        greedy: If True, use greedy decoding; otherwise beam search
        beam_width: Width of beam for beam search
        top_paths: Number of top paths to return
        
    Returns:
        Tuple of (decoded_sequences, log_probabilities)
    """
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
    input_length = tf.cast(input_length, dtype="int32")

    (decoded, log_prob) = tf.nn.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths,
    )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

@tf.keras.utils.register_keras_serializable(package="captchaResolver")
class CTCLayer(layers.Layer):
    """Custom CTC layer for computing CTC loss.
    
    This layer is registered as a serializable Keras layer to ensure
    proper model saving/loading without needing custom_objects in load_model.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Use local Keras3-compatible CTC batch cost implementation
        self.loss_fn = ctc_batch_cost
        self.supports_masking = True

    def get_config(self):
        """Return layer config for serialization."""
        config = super().get_config()
        return config

    def call(self, y_true, y_pred):
        """Compute CTC loss and return predictions."""
        batch_len = tf.shape(y_true)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        input_length = tf.fill([batch_len, 1], input_length)
        label_length = tf.fill([batch_len, 1], label_length)

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

class KerasModel:

    def __init__(self, train_data: TrainInfo, verbose=1, keras_model=True, saved_model=False):
        self.train_data = train_data
        self.char_to_num = layers.StringLookup(
            vocabulary=train_data.characters, mask_token=None, num_oov_indices=0
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.predict_model = None
        self.verbose = verbose
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

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len)[0][0][
            :, : self.train_data.label_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = (
                tf.strings.reduce_join(self.num_to_char(res + 1))
                .numpy()
                .decode("utf-8")
            )
            output_text.append(res)
        return output_text

    def load_prediction_model(self, model_path: str = None):

        if model_path is None:
            model_path = self.train_data.get_model_path()

        loaded = keras.models.load_model(model_path)
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
    ):
        """Train the model using the same logic previously in keras_engine.train_model.

        Returns:
            model_base_dir (str): path to the directory where the model was saved
        """
        # Prepare datasets
        train_dataset, validation_dataset = self.split_dataset(
            batch_size=batch_size, train_size=0.9, shuffle=True
        )

        train_data = self.train_data
        train_model = self.build_model()
        callbacks = []

        if earlystopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        # Fit
        if callbacks:
            train_model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=self.verbose,
            )
        else:
            train_model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                verbose=self.verbose,
            )

        # Save trained model
        model_base_dir = train_data.get_model_base_dir()
        full_model_path = os.path.join(model_base_dir, "weights.keras")
        train_model.save(full_model_path)
        return model_base_dir
