import os
import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class Config:
    IMG_SIZE = 380
    BATCH_SIZE = 12
    NUM_CLASSES = 5
    N_FOLDS = 10
    SEED = 42
    PHASE1_EPOCHS = 12
    PHASE2_EPOCHS = 20
    PHASE1_LR = 1e-3
    PHASE2_LR = 5e-5
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.05
    GRAD_CLIP = 1.0
    TTA_STEPS = 7
    BASE_PATH = "/kaggle/input/competitions/aptos2019-blindness-detection"
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
    TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
    CSV_PATH = os.path.join(BASE_PATH, "train.csv")
    TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")

cfg = Config()

np.random.seed(cfg.SEED)
tf.random.set_seed(cfg.SEED)
random.seed(cfg.SEED)

def load_and_preprocess_image(path):
    try:
        if isinstance(path, tf.Tensor):
            path = path.numpy().decode('utf-8')
        elif isinstance(path, bytes):
            path = path.decode('utf-8')
        elif not isinstance(path, str):
            path = str(path)

        if not os.path.exists(path):
            return np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)

        img = cv2.imread(path)
        if img is None:
            return np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)

        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords)
            if w > 10 and h > 10:
                img = img[y:y+h, x:x+w]

        img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        blur = cv2.GaussianBlur(img, (0, 0), 10)
        img = cv2.addWeighted(img, 4, blur, -4, 128)
        img = img.astype(np.float32) / 255.0
        return img
    except:
        return np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)

def tf_load_wrapper(path, label):
    img = tf.py_function(func=lambda p: load_and_preprocess_image(p), inp=[path], Tout=tf.float32)
    img.set_shape((cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    return img, label

def create_tf_dataset(paths, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(tf_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, label

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(cfg.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_model():
    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    backbone = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    x = backbone.output
    x = layers.Dropout(cfg.DROPOUT_RATE)(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.DROPOUT_RATE/2)(x)
    outputs = layers.Dense(4, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs, outputs)
    return model, backbone

def ordinal_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    thresholds = tf.range(4, dtype=tf.int32)
    y_true_expanded = tf.expand_dims(y_true, -1)
    ordinal_targets = tf.cast(thresholds < y_true_expanded, tf.float32)
    eps = cfg.LABEL_SMOOTHING
    ordinal_targets = ordinal_targets * (1 - eps) + (1 - ordinal_targets) * eps
    loss = tf.keras.losses.binary_crossentropy(ordinal_targets, y_pred)
    return tf.reduce_mean(loss)

def ordinal_accuracy(y_true, y_pred):
    pred_classes = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.int32), axis=1)
    y_true = tf.cast(y_true, tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(pred_classes, y_true), tf.float32))