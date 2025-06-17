



# import LoadBatches1D_1
from keras import Input, Model
from tensorflow import keras
from tensorflow.keras import optimizers, activations
import warnings
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

# Disable TensorFlow v2 behavior (use v1 style)
tf.compat.v1.disable_v2_behavior()

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCR_GPU_ALLOW_GROWTH"] = "true"  # Try allocating GPU memory as needed

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
exp_id = 'FE30_test'  # Experiment ID
# Class weights for 4-class task
weights = [0.5, 1.5]  # Adjust weights based on actual data
# weights = [0.5, 2]

# Define weighted categorical crossentropy loss function suitable for 4-class classification
def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        weighted_losses = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(weighted_losses, axis=-1)
        return tf.reduce_mean(loss)

    return loss

def lr_schedule(epoch):
    # Learning rate schedule
    lr = 0.00005
    print('Learning rate: ', lr)
    return lr

# Directory for training and validation sets
train_sigs_path = '../data/train_data/'
train_segs_path = '../data/train_label/'
val_sigs_path = '../data/val_data/'
val_segs_path = '../data/val_label/'  # Validation label path

# Data preprocessed by experts, removing full-day infrastructure noise
SAVE_DIR = '../image/image_jijian' + '/{}'.format(exp_id)
if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))
print("SAVE_DIR", SAVE_DIR)

train_batch_size = 8  # Reduce batch size for weak local GPU
n_classes = 2  # Change from 3 to 2
input_length = 1440  # In seconds; was 1800, now 1440; full day = 86400
optimizer_name = optimizers.Adam(lr_schedule(0), clipnorm=1)
PATIENCE = 20  # Early stopping patience
val_batch_size = 32
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
input_shape = (1440, 3)
num_classes = 2

# Build model
from CNN_Transformer import build_model
model = build_model()

model.compile(loss=weighted_categorical_crossentropy(weights),
              optimizer=optimizer_name,
              metrics=['accuracy'])

model.summary()

print("Experiment code = {}".format(exp_id))
print("PATIENCE = {}".format(PATIENCE))

output_length = 1440
import LoadBatches1D
G = LoadBatches1D.SigSegmentationGenerator(train_sigs_path, train_segs_path, train_batch_size, n_classes, output_length)
G2 = LoadBatches1D.SigSegmentationGenerator(val_sigs_path, val_segs_path, val_batch_size, n_classes, output_length)

# Callbacks: checkpoint & early stopping
checkpointer = [keras.callbacks.ModelCheckpoint(filepath='{}/bmodel.h5'.format(SAVE_DIR), monitor='val_acc', mode='max', save_best_only=True),
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE)]

# Training
history = model.fit_generator(G, 991 // train_batch_size,
                              validation_data=G2,
                              validation_steps=int(330 / val_batch_size),
                              epochs=100,
                              callbacks=[checkpointer, lr_scheduler])

# Plot Accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history.get('val_accuracy') or history.history.get('val_acc'))
plt.title('Model accuracy {}'.format(exp_id))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)
plt.savefig('{}/Accuracy.png'.format(SAVE_DIR))  

# Save accuracy log
print("Saving accuracy file")
txt_name2 = "acc_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name2, "w")
this_file.write("acc")
this_file.write(str(history.history['acc']))
this_file.write("\n")
this_file.write("val_acc")
this_file.write(str(history.history['val_acc']))
this_file.write("\n")
this_file.close()
print("END of accuracy saving")

# Plot Loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss {}'.format(exp_id))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)
plt.savefig('{}/Loss.png'.format(SAVE_DIR))

# Save loss log
print("Saving loss file")
txt_name = "loss_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name, "w")
this_file.write("loss")
this_file.write(str(history.history['loss']))
this_file.write("\n")
this_file.write("val_loss")
this_file.write(str(history.history['val_loss']))
this_file.write("\n")
this_file.close()

print("Accuracy & Loss plots saved. Check folder: {}".format(SAVE_DIR))
print("Best model saved to: {}".format(SAVE_DIR))



#test
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
import LoadBatches1D_1

test_geo_path = '../data/test_data/'  # Test dataset path
test_label_path = '../data/test_label/'  # Test label path

test_geo_files = os.listdir(test_geo_path)
test_label_files = os.listdir(test_label_path)

# Weighted categorical crossentropy loss function
def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return loss
    return loss

weights = [0.5, 2]
custom_loss = weighted_categorical_crossentropy(weights)
num_classes = 2

SAVE_DIR = '../image/image_jijian' + '/{}'.format(exp_id)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
print("SAVE_DIR", SAVE_DIR)

from tensorflow.keras.models import load_model
from CNN_Transformer import SwinTransformerBlock, AdaptiveFusionBlock, CATM
from wtconv1 import WTConv1D
from FFT import FourierUnit
from FFTBlock import FFTNetBlock
from mamba1 import Mamba, MambaBlock
from HAAM import HAAM
from u2net import REBNCONV

best_model_path = '{}/bmodel.h5'.format(SAVE_DIR)
model = load_model(best_model_path, compile=False, custom_objects={
    'weighted_categorical_crossentropy': custom_loss, 'WTConv1D': WTConv1D,
    'FourierUnit': FourierUnit, 'CATM': CATM,
    'SwinTransformerBlock': SwinTransformerBlock,
    'AdaptiveFusionBlock': AdaptiveFusionBlock,
    'FFTNetBlock': FFTNetBlock, 'Mamba': Mamba,
    'MambaBlock': MambaBlock, 'HAAM': HAAM,
    'REBNCONV': REBNCONV
})

label_test = []
label_pred = []
probabilities_hvdc = []
probabilities_background = []

model.compile(loss=weighted_categorical_crossentropy(weights), optimizer='adam', metrics=['accuracy'])

intersection_hvdc = 0
union_hvdc = 0
intersection_background = 0
union_background = 0

from sklearn import preprocessing as prep

if len(test_geo_files) == len(test_label_files):
    for i in range(len(test_geo_files)):
        select = test_geo_files[i]
        if select == test_label_files[i]:
            a_geo = np.load(test_geo_path + select)
            a_label = np.load(test_label_path + select)
            a_label_list = a_label.tolist()
            label_test.extend(a_label_list)

            a_geo = np.expand_dims(a_geo, axis=0)
            a_pred = model.predict(a_geo, batch_size=4)

            a_pred_array = []
            for j in range(len(a_pred[0])):
                prob_background = a_pred[0][j][0]
                prob_hvdc = a_pred[0][j][1]
                probabilities_hvdc.append(prob_hvdc)
                probabilities_background.append(prob_background)
                pred_label = 1.0 if prob_hvdc >= 0.5 else 0.0
                a_pred_array.append(pred_label)

                real_label = a_label[j]
                if real_label == 1.0 and pred_label == 1.0:
                    intersection_hvdc += 1
                if real_label == 1.0 or pred_label == 1.0:
                    union_hvdc += 1
                if real_label == 0.0 and pred_label == 0.0:
                    intersection_background += 1
                if real_label == 0.0 or pred_label == 0.0:
                    union_background += 1

            label_pred.extend(a_pred_array)
            if i % 100 == 0:
                print("{}/{}".format(i + 1, len(test_geo_files)))
        else:
            print("File names do not match.")
            break

# Print metrics
print("len true", len(label_test))
print("len pred", len(label_pred))
acc_nums = accuracy_score(label_test, label_pred)
print("acc_nums", acc_nums)
print("{}".format(exp_id))
acc = accuracy_score(label_test, label_pred)
print("acc", acc)
p = precision_score(label_test, label_pred, zero_division=0)
print("precision", p)
r = recall_score(label_test, label_pred, zero_division=0)
print("recall", r)
f1 = f1_score(label_test, label_pred, zero_division=0)
print("f1", f1)

iou_hvdc = intersection_hvdc / union_hvdc if union_hvdc > 0 else 0
iou_background = intersection_background / union_background if union_background > 0 else 0
miou = (iou_hvdc + iou_background) / 2
print("IoU HVDC:", iou_hvdc)
print("IoU Background:", iou_background)
print("mIoU:", miou)
auc = roc_auc_score(label_test, probabilities_hvdc)
print("AUC: {:.4f}".format(auc))

# Classification report
acc = accuracy_score(label_test, label_pred)
print("Overall Accuracy:", acc)
precision_per_class = precision_score(label_test, label_pred, average=None, zero_division=0)
recall_per_class = recall_score(label_test, label_pred, average=None, zero_division=0)
f1_per_class = f1_score(label_test, label_pred, average=None, zero_division=0)

classes = np.unique(label_test)
print("\nPer-Class Metrics (Including Accuracy):")
for cls in classes:
    cls_mask = (np.array(label_test) == cls)
    cls_true = np.array(label_test)[cls_mask]
    cls_pred = np.array(label_pred)[cls_mask]
    cls_accuracy = np.mean(cls_true == cls_pred) if len(cls_true) > 0 else 0.0
    print(f"Class {cls}: Accuracy: {cls_accuracy:.4f}, "
          f"Precision: {precision_per_class[int(cls)]:.4f}, "
          f"Recall: {recall_per_class[int(cls)]:.4f}, "
          f"F1 Score: {f1_per_class[int(cls)]:.4f}")

precision_macro = precision_score(label_test, label_pred, average='weighted', zero_division=0)
recall_macro = recall_score(label_test, label_pred, average='weighted', zero_division=0)
f1_macro = f1_score(label_test, label_pred, average='weighted', zero_division=0)
print("\nPer-Class Metrics:")
for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
    print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}")

print("\nOverall Metrics:")
print(f"Overall Precision (Macro): {precision_macro:.4f}")
print(f"Overall Recall (Macro): {recall_macro:.4f}")
print(f"Overall F1 Score (Macro): {f1_macro:.4f}")
print("\nClassification Report:")
print(classification_report(label_test, label_pred, zero_division=0))

# Compute per-class IoU again using confusion matrix
conf_matrix = confusion_matrix(label_test, label_pred)
iou_per_class = []
for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    iou_per_class.append(iou)
miou = np.mean(iou_per_class)
print("\nPer-Class IoU:")
for i, iou in enumerate(iou_per_class):
    print(f"Class {i}: IoU: {iou:.4f}")
print(f"mIoU: {miou:.4f}")
