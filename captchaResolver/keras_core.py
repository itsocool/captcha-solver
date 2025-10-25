import os
import shutil
import numpy as np
import keras
import tensorflow as tf
from typing import Optional, Tuple, List, Union
from keras import layers, models, ops, saving
from captchaResolver.dataclass import CaptchaType, TrainData

def ctc_decode(
    y_pred: tf.Tensor,
    input_length: Union[tf.Tensor, np.ndarray],
    greedy: bool = True,
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
        # Greedy decoding: Fast, picks most probable class at each timestep
        decoded, log_probabilities = tf.nn.ctc_greedy_decoder(
            inputs=y_pred_log,
            sequence_length=input_length,
            merge_repeated=True  # Collapse repeated characters (CTC behavior)
        )
    else:
        # Beam search: Explores multiple candidate paths for better accuracy
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            inputs=y_pred_log,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    
    # Convert sparse tensors to dense format for easier handling
    decoded_dense = []
    for sparse_tensor in decoded:
        # Reconstruct sparse tensor with correct shape
        reconstructed = tf.SparseTensor(
            indices=sparse_tensor.indices,
            values=sparse_tensor.values,
            dense_shape=(batch_size, time_steps)
        )
        # Convert to dense, using -1 for padding
        dense_tensor = tf.sparse.to_dense(
            sp_input=reconstructed,
            default_value=-1
        )
        decoded_dense.append(dense_tensor)
    
    return decoded_dense, log_probabilities

@saving.register_keras_serializable(package="captchaResolver")
class CTCLayer(layers.Layer):
    """Custom Keras layer for CTC (Connectionist Temporal Classification) loss.
    
    This layer computes CTC loss during training and acts as a pass-through
    during inference. It's designed to be inserted between the model's output
    and the training objective, automatically handling the loss computation.
    
    The layer is registered as serializable using Keras 3.x API, ensuring
    proper model saving and loading without requiring custom_objects parameter.
    
    Architecture:
        - Training: Computes CTC loss and adds it to the model's total loss
        - Inference: Returns predictions unchanged (identity function)
    
    Args:
        from_logits: Whether model outputs raw logits (True) or softmax 
                    probabilities (False). 
                    - True: More numerically stable (recommended)
                    - False: Standard approach, expects softmax output
                    Default: False
        blank_index: Index of the CTC blank token in the output vocabulary.
                    - Use -1 for last position (most common)
                    - Use 0 for first position
                    - Or specify exact index
                    The blank represents "no character" in CTC alignment.
                    Default: -1
        name: Layer name for identification in model architecture.
              Default: None (auto-generated)
        **kwargs: Additional keyword arguments passed to base Layer class.
    
    Attributes:
        from_logits: Stored logits mode setting
        blank_index: Stored blank token index
        supports_masking: Always True (CTC handles variable lengths)
    
    Example:
        >>> # Build training model with CTC loss
        >>> inputs = layers.Input(shape=(None, features))
        >>> x = layers.LSTM(128, return_sequences=True)(inputs)
        >>> outputs = layers.Dense(num_classes, activation='softmax')(x)
        >>> 
        >>> # Add CTC layer for training
        >>> labels = layers.Input(shape=(None,), dtype='float32')
        >>> ctc_output = CTCLayer()(labels, outputs)
        >>> 
        >>> # Compile model
        >>> model = keras.Model(inputs=[inputs, labels], outputs=ctc_output)
        >>> model.compile(optimizer='adam')  # No loss needed, handled by CTCLayer
        >>> 
        >>> # For inference, use model without CTC layer
        >>> inference_model = keras.Model(inputs=inputs, outputs=outputs)
    
    Notes:
        - The layer expects two inputs: (labels, predictions)
        - Loss is automatically added via self.add_loss()
        - No explicit loss function needed in model.compile()
        - For best results, use from_logits=True and remove final softmax
    """
    
    def __init__(
        self,
        from_logits: bool = False,
        blank_index: int = -1,
        name: Optional[str] = None,
        **kwargs
    ):
        """Initialize CTC layer with loss configuration.
        
        Args:
            from_logits: Whether inputs are logits (True) or probabilities (False)
            blank_index: Index of blank token (-1 for last position)
            name: Optional layer name
            **kwargs: Additional layer arguments
        """
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
        """Compute CTC loss and return predictions unchanged.
        
        This method is called during both training and inference:
        - Training: Computes loss and adds it to model.losses
        - Inference: Returns predictions as-is (identity operation)
        
        Args:
            y_true: Ground truth labels with shape (batch, max_label_len).
                   Integer tensor containing character indices.
                   Positions beyond actual label length are padding.
            y_pred: Model predictions with shape (batch, time, num_classes).
                   - If from_logits=False: softmax probabilities
                   - If from_logits=True: raw logits
            training: Optional training mode flag. Currently unused but included
                     for future extensions (e.g., different behavior in train/test).
        
        Returns:
            y_pred: Predictions unchanged (pass-through).
                   Shape: (batch, time, num_classes)
        
        Side Effects:
            Adds CTC loss to self.losses during training via self.add_loss().
            This loss is automatically included in the model's total loss.
        """
        # Extract dimensions from input tensors
        batch_size = ops.shape(y_true)[0]
        time_steps = ops.shape(y_pred)[1]
        max_label_len = ops.shape(y_true)[1]

        # Create length tensors
        # All samples in batch have same time_steps (model output length)
        input_length = ops.cast(
            ops.repeat(time_steps, batch_size),
            dtype=tf.int32
        )
        input_length = ops.reshape(input_length, (batch_size, 1))
        
        # All samples have same max_label_len (label tensor width)
        label_length = ops.cast(
            ops.repeat(max_label_len, batch_size),
            dtype=tf.int32
        )
        label_length = ops.reshape(label_length, (batch_size, 1))

        # Compute CTC loss using configured parameters
        loss = self.ctc_batch_cost(
            y_true=y_true,
            y_pred=y_pred,
            input_length=input_length,
            label_length=label_length,
            from_logits=self.from_logits,
            blank_index=self.blank_index
        )
        
        # Add loss to model's total loss
        # Keras will automatically include this in training updates
        self.add_loss(loss)

        # Return predictions unchanged (identity for inference)
        return y_pred

    def ctc_label_dense_to_sparse(
        self,
        labels: tf.Tensor, 
        label_lengths: tf.Tensor
    ) -> tf.SparseTensor:
        """Convert dense labels to sparse format for CTC.
        
        This function efficiently converts dense integer labels into the sparse tensor
        format required by TensorFlow's CTC loss implementation. It uses vectorized
        operations for optimal memory usage and performance.
        
        Compatible with TensorFlow 2.x and Keras 3.x.
        
        Args:
            labels: Dense integer labels tensor with shape (batch_size, max_label_len).
                    Values should be non-negative integers representing character indices.
                    Padding values (beyond label_lengths) are ignored.
            label_lengths: Integer tensor with shape (batch_size,) indicating the actual
                        length of each label sequence (excluding padding).
        
        Returns:
            tf.SparseTensor: Sparse representation of the labels with:
                - indices: 2D tensor of shape (num_valid_labels, 2) containing [batch_idx, label_idx]
                - values: 1D tensor containing the label values at valid positions
                - dense_shape: Shape of the original dense tensor [batch_size, max_label_len]
        
        Example:
            >>> labels = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.int32)
            >>> label_lengths = tf.constant([3, 2], dtype=tf.int32)
            >>> sparse = ctc_label_dense_to_sparse(labels, label_lengths)
            >>> # sparse.values = [1, 2, 3, 4, 5]
        """
        # Get tensor dimensions
        label_shape = ops.shape(labels)
        batch_size = label_shape[0]
        max_label_len = label_shape[1]

        # Create mask using vectorized broadcasting
        # This approach is more efficient than iterating through batches
        label_positions = ops.arange(max_label_len)  # [0, 1, 2, ..., max_label_len-1]
        label_lengths_expanded = ops.expand_dims(label_lengths, axis=1)  # (batch_size, 1)
        
        # Broadcasting comparison creates a boolean mask
        # True where position < actual label length, False for padding
        valid_positions_mask = label_positions < label_lengths_expanded  # (batch_size, max_label_len)

        # Extract indices of valid (non-padding) positions
        # Returns 2D array where each row is [batch_idx, position_idx]
        valid_indices = tf.where(valid_positions_mask)

        # Gather actual label values at valid positions
        valid_label_values = tf.gather_nd(labels, valid_indices)

        # Construct and return sparse tensor
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
        """Normalize input and label lengths for CTC operations.
        
        Ensures lengths are in the correct shape and dtype for CTC functions.
        Handles both (batch,) and (batch, 1) shaped inputs.
        
        Args:
            input_length: Input sequence lengths, shape (batch,) or (batch, 1)
            label_length: Label sequence lengths, shape (batch,) or (batch, 1)
        
        Returns:
            Tuple of (normalized_input_length, normalized_label_length),
            both with shape (batch,) and dtype int32
        """
        # Squeeze to ensure (batch,) shape and cast to int32
        label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype=tf.int32)
        input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype=tf.int32)
        
        return input_length, label_length

    def _convert_to_time_major_logits(
        self,
        y_pred: tf.Tensor,
        from_logits: bool = False
    ) -> tf.Tensor:
        """Convert predictions to time-major log-probabilities.
        
        Handles both logits and probabilities, converting to the time-major
        log-probability format expected by TF's CTC implementation.
        
        Args:
            y_pred: Predictions with shape (batch, time, num_classes)
            from_logits: Whether y_pred contains raw logits (True) or 
                        softmax probabilities (False)
        
        Returns:
            Log-probabilities with shape (time, batch, num_classes)
        """
        # Transpose from (batch, time, classes) to (time, batch, classes)
        y_pred_time_major = ops.transpose(y_pred, axes=[1, 0, 2])
        
        if from_logits:
            # Already logits, use directly (will be passed to loss function)
            return y_pred_time_major
        else:
            # Convert probabilities to log-probabilities
            # Add epsilon for numerical stability (avoid log(0))
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
        """Compute CTC (Connectionist Temporal Classification) batch loss.
        
        This implementation is fully compatible with TensorFlow 2.x and Keras 3.x,
        supporting both raw logits and softmax probabilities. It efficiently handles
        variable-length sequences through the CTC algorithm.
        
        The CTC loss allows training sequence models without requiring aligned
        input-output pairs, making it ideal for tasks like OCR where character
        positions in the input image are unknown.
        
        Args:
            y_true: Ground truth labels, dense integer tensor with shape 
                    (batch, max_label_len). Values are character indices.
                    Positions beyond label_length are ignored (padding).
            y_pred: Model predictions with shape (batch, time, num_classes).
                    - If from_logits=False: expects softmax probabilities
                    - If from_logits=True: expects raw logits (recommended)
            input_length: Actual length of each input sequence, shape (batch,) or (batch, 1).
                        Specifies how many time steps are valid for each sample.
            label_length: Actual length of each label sequence, shape (batch,) or (batch, 1).
                        Specifies how many characters are in each label.
            from_logits: If True, y_pred contains raw logits (more numerically stable).
                        If False, y_pred contains softmax probabilities.
                        Default: False (for backward compatibility).
            blank_index: Index of the CTC blank token. Use -1 for last index (recommended).
                        The blank token represents "no character" in CTC alignment.
                        Default: -1
        
        Returns:
            CTC loss tensor with shape (batch, 1). Each element contains the
            negative log-likelihood of the corresponding sequence.
        
        Raises:
            tf.errors.InvalidArgumentError: If input dimensions are incompatible
            
        Example:
            >>> # Training setup
            >>> y_true = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.int32)
            >>> y_pred = model(images)  # shape: (2, 10, 6)
            >>> input_length = tf.constant([[10], [10]], dtype=tf.int32)
            >>> label_length = tf.constant([[3], [2]], dtype=tf.int32)
            >>> loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)
            >>> # loss shape: (2, 1)
        
        Notes:
            - For best numerical stability, use from_logits=True and output raw logits
            from your model (remove final softmax activation).
            - The blank_index=-1 means the blank token is at index (num_classes - 1).
            - CTC automatically handles alignment and repeated characters.
        """
        # Normalize length tensors to consistent shape and dtype
        input_length, label_length = self._normalize_ctc_inputs(input_length, label_length)

        # Convert dense labels to sparse format (required by tf.nn.ctc_loss)
        sparse_labels = ops.cast(
            self.ctc_label_dense_to_sparse(y_true, label_length),
            dtype=tf.int32
        )

        # Convert predictions to time-major log-probabilities
        logits = self._convert_to_time_major_logits(y_pred, from_logits=from_logits)

        # Compute CTC loss using TensorFlow's optimized implementation
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=logits,
            label_length=None,  # Already encoded in sparse_labels structure
            logit_length=input_length,
            logits_time_major=True,  # We transposed to time-major format
            blank_index=blank_index,
        )
        
        # Reshape to (batch, 1) for consistency with Keras loss expectations
        return ops.expand_dims(loss, axis=1)

class KerasModel:

    def __init__(self, captcha_type: CaptchaType, verbose=1, keras_model=True, saved_model=False):
        self.captcha_type: CaptchaType = captcha_type
        self.train_data: TrainData = captcha_type.train_data
        self.char_to_num = layers.StringLookup(
            vocabulary=self.train_data.characters, mask_token=None, num_oov_indices=0
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

    def decode_batch_predictions(self, pred, use_greedy=True):
        """Decode batch predictions to text strings.
        
        Args:
            pred: Prediction probabilities, shape (batch, time, num_classes)
            use_greedy: If True, use greedy decoding; if False, use beam search
        
        Returns:
            List of decoded text strings
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Decode using CTC decoder (greedy or beam search)
        results = ctc_decode(
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
        """Train the model with automatic checkpointing and final model saving.
        
        During training:
        - Best model (based on val_loss) is saved to 'best_weights.keras'
        - Training can be stopped early if no improvement for patience epochs
        
        After training:
        - Best model is copied to 'weights.keras' as the final model
        - Temporary best model file is removed to save disk space
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            earlystopping: Whether to use early stopping callback
            early_stopping_patience: Epochs to wait before stopping if no improvement
        
        Returns:
            model_base_dir (str): Path to the directory where models were saved
        """
        # Prepare datasets
        train_dataset, validation_dataset = self.split_dataset(
            batch_size=batch_size, train_size=0.9, shuffle=True
        )

        train_data = self.train_data
        train_model = self.build_model()
        
        # Setup model save directory
        model_base_dir = train_data.get_model_base_dir()
        os.makedirs(model_base_dir, exist_ok=True)
        
        # Define checkpoint path for best model
        best_model_path = os.path.join(model_base_dir, "best_weights.keras")
        
        # Setup callbacks
        callbacks = []
        
        # ModelCheckpoint: Save best model during training
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
        
        # EarlyStopping with patience counter display
        if earlystopping:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=False,  # We'll load from checkpoint instead
                verbose=0,  # Disable default verbose, use custom callback
            )
            callbacks.append(early_stopping_callback)
            
            # Custom callback to display remaining patience
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

        # Copy best model to final model path and cleanup
        final_model_path = os.path.join(model_base_dir, "weights.keras")
        
        if os.path.exists(best_model_path):
            # Copy best model to final location
            shutil.copy2(best_model_path, final_model_path)
            
            # Remove temporary best model file to save disk space
            try:
                os.remove(best_model_path)
                if self.verbose:
                    print(f"\n✓ 학습 완료:")
                    print(f"  - 최종 모델: {final_model_path}")
                    print(f"  - 임시 파일 정리 완료 (best_weights.keras 삭제됨)")
            except OSError as e:
                if self.verbose:
                    print(f"\n✓ 학습 완료:")
                    print(f"  - 최종 모델: {final_model_path}")
                    print(f"  ⚠ 임시 파일 정리 실패: {e}")
        else:
            # Fallback: save current model if checkpoint doesn't exist
            train_model.save(final_model_path)
            if self.verbose:
                print(f"\n✓ 학습 완료:")
                print(f"  - 최종 모델: {final_model_path}")
                print(f"  ⚠ best model이 생성되지 않았습니다 (현재 모델 저장됨)")
        
        return model_base_dir

    def batch_predict(
        self, batch_size=32
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
            preds = model.decode_batch_predictions(pred_vals)
            for label in labels:
                label_text = tf.strings.reduce_join(
                    model.num_to_char(label + 1)
                ).numpy().decode("utf-8")
                all_labels.append(label_text)
            
            all_preds.extend(preds)
            # Compute a single confidence value per sample.
            # pred_vals shape: (batch, time, classes)
            # 1) For each timestep, take the max class probability -> shape (batch, time)
            # 2) Average across time to get a scalar confidence per sample -> shape (batch,)
            # 3) Multiply by 100 to present as percentage (matching prints elsewhere)
            batch_confidences = np.mean(np.max(pred_vals, axis=2), axis=1) * 100.0
            # Ensure plain Python floats (avoid numpy types / lists that break formatting)
            all_confidences.extend([float(c) for c in batch_confidences.tolist()])
        return all_preds, all_labels, all_confidences, pred_img_path_list

    def predict(
        self,
        image_path: str,
        model_path: str = None,
        use_greedy: bool = False,
    ) -> Tuple[str, float]:
        if model_path is not None:
            self.load_prediction_model(model_path=model_path)
        elif self.predict_model is None:
            self.load_prediction_model()
        sample = self.encode_single_sample(image_path)
        image = ops.expand_dims(sample["image"], axis=0)
        pred = self.predict_model.predict(image)
        decoded_preds = self.decode_batch_predictions(pred, use_greedy=use_greedy)
        confidence = np.max(pred)
        return decoded_preds[0], confidence
    