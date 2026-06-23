import platform
from os.path import join
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from deepmreye import architecture
from deepmreye.util import data_generator, util
from tensorflow.keras.callbacks import EarlyStopping

# Function copied from deepmreye architecture, I don't know why but it didn't see this function so I had to copy it
def get_adam_optimizer(learning_rate):
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() in ["arm64", "aarch64"]

    if is_mac and is_arm:
        # Apple Silicon detected
        from keras.optimizers.legacy import Adam
        print("Apple Silicon detected - using legacy Adam optimizer.")

    else:
        from keras.optimizers import Adam

    return Adam(learning_rate=learning_rate)


def train_model_with_smoothl1(
    dataset,
    generators,
    opts,
    is_resting_state=False,
    clear_graph=True,
    save=False,
    model_path="./",
    workers=4,
    use_multiprocessing=True,
    models=None,
    return_untrained=False,
    verbose=0,
    pretrained_weights=None,      
    freeze_backbone=True,  
):

    """Function train with Smooth L1 option"""

    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() in ["arm64", "aarch64"]
    if is_mac and is_arm and use_multiprocessing:
        print("Apple Silicon detected - Multiprocessing is not supported. Disabling it.")
        use_multiprocessing = False

    if clear_graph:
        K.clear_session()

    if use_multiprocessing:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        workers = 1

    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators

    ((X, y), _) = next(training_generator)

    # Learning rate scheduler

    lr_sched = util.step_decay_schedule(
        initial_lr=opts["lr"], 
        decay_factor=0.9, 
        num_epochs=opts["epochs"]
    )

    early_stop_cb = EarlyStopping(
        monitor='val_smoothl1_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )

    callbacks_list = [lr_sched, early_stop_cb]

    if models is None:
        if is_resting_state:
            model, model_inference = create_model_with_smoothl1(
                X.shape[1::], 
                opts,
                is_resting_state=True
            )

        else:
            model, model_inference = architecture.create_standard_model(
                X.shape[1::], 
                opts
            )
    else:
        model, model_inference = models

    if pretrained_weights is not None:
        print(f"Loading pretrained weights: {pretrained_weights}")
        model_inference.load_weights(pretrained_weights)

        if freeze_backbone:
            print("🧊 Freezing backbone layers")
            for layer in model.layers:
                if "fc" not in layer.name and "confidence" not in layer.name:
                    layer.trainable = False
            optimizer = get_adam_optimizer(opts["lr"])
            model.compile(optimizer=optimizer)

    if return_untrained:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        return (model, model_inference)

    if verbose > 1:
        print(model.summary(line_length=200))

    history = model.fit(
        training_generator,
        steps_per_epoch=opts["steps_per_epoch"],
        epochs=opts["epochs"],
        validation_data=testing_generator,
        validation_steps=opts["validation_steps"],
        callbacks=callbacks_list, 
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        verbose = verbose
    )

    if save:
        model_inference.save_weights(
            join(model_path, f"modelinference_{dataset}.h5")
        )

    

    if use_multiprocessing:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    return (model, model_inference, history)



def pearson_loss_tf(y_true, y_pred, eps=1e-8):
    y_true = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)

    numerator = tf.reduce_sum(y_true * y_pred, axis=0)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(y_true), axis=0) *
        tf.reduce_sum(tf.square(y_pred), axis=0)
    ) + eps

    r = numerator / denominator
    return 1.0 - tf.reduce_mean(r)


def create_model_with_smoothl1(input_shape, opts, is_resting_state=False):
    """Creates model with Smooth L1 loss for resting state"""

    base_model, model_inference = architecture.create_standard_model(
        input_shape,
        opts
    )
    if not is_resting_state:
        return base_model, model_inference
    print("\nUsing Smooth L1 loss for resting state data\n")


    # Smooth L1 loss
    def smooth_l1(y_true, y_pred):
        delta = opts.get("smooth_l1_delta", 1.0)
        error = y_true - y_pred
        abs_error = tf.abs(error)

        return tf.where(
            abs_error < delta,
            0.5 * tf.square(error),
            delta * (abs_error - 0.5 * delta)
        )

    inputs = base_model.inputs                       # [image_input, regression_target]
    out_regression = model_inference.outputs[0]      # regresja
    out_confidence = model_inference.outputs[1]      # confidence
    real_regression = inputs[1]                      # ground truth

    euclidean_loss = tf.sqrt(tf.reduce_sum(
        smooth_l1(real_regression, out_regression),
        axis=-1
    ))

    confidence_loss = tf.square(euclidean_loss - out_confidence)
    pearson_loss = pearson_loss_tf(real_regression, out_regression)

    train_model = tf.keras.Model(
        inputs=inputs,
        outputs=[],        
        name="smoothl1_training_model"
    )



    train_model.add_loss(opts["loss_euclidean"] * tf.reduce_mean(euclidean_loss))
    train_model.add_loss(opts["loss_confidence"] * tf.reduce_mean(confidence_loss))
    train_model.add_loss(opts.get("loss_pearson", 0.1) * pearson_loss)


    train_model.add_metric(tf.reduce_mean(euclidean_loss), name="smoothl1_loss")
    train_model.add_metric(tf.reduce_mean(confidence_loss), name="confidence_loss")
    train_model.add_metric(1.0 - pearson_loss, name="pearson_r")

    optimizer = get_adam_optimizer(opts["lr"])
    train_model.compile(optimizer=optimizer)

    print("Smooth L1 model compiled correctly.\n")
    return train_model, model_inference