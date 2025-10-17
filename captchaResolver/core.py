import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import captchaResolver.dataclass as dataclass

def get_captcha_type_list(image_dir: str = "./images", model_dir: str = "./model"):
    default = dataclass.CaptchaType(id="default", name="기본 캡챠", desc="기본 캡챠")
    supreme_court = dataclass.CaptchaType(id="supreme_court", name="대법원", desc="대법원 캡챠")
    gov24 = dataclass.CaptchaType(id="gov24", name="gov24", desc="대한민국 정부 24 캡챠")
    wetax = dataclass.CaptchaType(id="wetax", name="wetax", desc="WETAX 캡챠")
    kshop = dataclass.CaptchaType(id="kshop", name="kshop", desc="KT Shopping 캡챠")

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
        "kshop": kshop,
    }

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """CTC loss function using TF 2.x native API."""
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    # Use TF 2.x native ctc_loss instead of deprecated compat.v1
    return ops.expand_dims(
        tf.nn.ctc_loss(
            labels=sparse_labels, 
            logits=y_pred, 
            label_length=None,
            logit_length=input_length,
            blank_index=-1
        ),
        1,
    )

def ctc_label_dense_to_sparse(labels, label_lengths):
    """Convert dense labels to sparse format for CTC."""
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    # Replace deprecated tf.compat.v1.scan with tf.scan
    dense_mask = tf.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"), 
        vals_sparse, 
        ops.cast(label_shape, dtype="int64")
    )

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
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
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        # Use TF 2.x native beam search decoder
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

@keras.saving.register_keras_serializable(package="captchaResolver")
class CTCLayer(layers.Layer):
    """Custom CTC layer for computing CTC loss.
    
    This layer is registered as a serializable Keras layer to ensure
    proper model saving/loading without needing custom_objects in load_model.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
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

    def __init__(self, train_data: dataclass.TrainInfo, hard_mode=False, keras_native=True, verbose=1):
        self.train_data = train_data
        self.char_to_num = layers.StringLookup(
            vocabulary=train_data.characters, mask_token=None, num_oov_indices=0
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.hard_mode = hard_mode
        self.keras_native = keras_native
        self.predict_model = None
        self.verbose = verbose

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
            validation_dataset.map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return train_dataset, validation_dataset

    def encode_single_sample(self, image_path, label):
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

        label = self.char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8")
        )
        return {"image": image, "label": label}

    def build_model(self):
        # Inputs to the model
        width, height = self.train_data.image_width, self.train_data.image_height
        input_img = layers.Input(shape=(width, height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        if self.hard_mode :
            x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), name="pool1")(x)

            x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), name="pool2")(x)

            new_shape = (
                (self.train_data.image_width // 4),
                (self.train_data.image_height // 4) * 128,
            )
            x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
            x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", name="dense1")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

            x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.9
            )

            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        else:

            x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
            x = layers.MaxPooling2D((2, 2), name="pool1")(x)
            x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
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

            optimizer = keras.optimizers.Adam()
        
        # Output layer
        unit = len(list(self.train_data.characters)) + 1
        x = layers.Dense(unit, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")

        # Compile the model and return
        model.compile(optimizer=optimizer)
        return model

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
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

    def load_prediction_model(self):

        # if self.weights_only:
        #     model = self.build_model(hard_mode=True)
        #     weights_path = self.train_data.get_model_path(weights_only=True)
        #     model.load_weights(weights_path)
        # else:
        model_path = self.train_data.get_model_path(keras_native=self.keras_native)

        if self.keras_native:
            # Pass custom_objects to ensure the custom CTCLayer is found during
            # deserialization. This is a robust fallback in case registry-based
            # lookup does not locate the class.
            model:keras.models.Model = keras.models.load_model(
                model_path, custom_objects={"CTCLayer": CTCLayer}
            )
            # model:keras.models.Model = keras.models.load_model(model_path)
        else:
            model:keras.models.Model = tf.saved_model.load(model_path)

        input_layer = model.input[0]
        output_layer = model.get_layer(name="dense2").output

        self.predict_model = keras.models.Model(
            input_layer, output_layer
        )

        return self.predict_model
