from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.losses as ls
import tensorflow.keras.optimizers as op
import tensorflow.keras.callbacks as tfc
import tensorflow as tf
import Model
import json

IMG_SIZE=(224,224)
BATCH_SIZE=32
LEARNING_RATE_OUTPUT=0.0001
LEARNING_RATE_TUNE=0.00001
START_LAYER=125
PATIENCE=6
model=Model.model
base_model=Model.base_model
train=image_dataset_from_directory(
    "datasets/fanconic/skin-cancer-malignant-vs-benign/versions/4/train",
    image_size=IMG_SIZE,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    seed=56,
    subset="training"


)
validation=image_dataset_from_directory(
    "datasets/fanconic/skin-cancer-malignant-vs-benign/versions/4/train",
    image_size=IMG_SIZE,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    seed=56,
    subset="validation"



)
test=image_dataset_from_directory(
    "datasets/fanconic/skin-cancer-malignant-vs-benign/versions/4/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE


)
callback=tfc.EarlyStopping(patience=PATIENCE,restore_best_weights=True)
model.compile(optimizer=op.Adam(learning_rate=LEARNING_RATE_OUTPUT),loss=ls.BinaryCrossentropy(),metrics=["accuracy", "precision", "recall", "auc", "true_positives", "true_negatives", "false_positives", "false_negatives"]
)
history1=model.fit(train,validation_data=validation,epochs=50,callbacks=[callback])
base_model.trainable = True
for layer in base_model.layers[:START_LAYER]:
    layer.trainable=False
for layer in base_model.layers[START_LAYER:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model.compile(optimizer=op.Adam(learning_rate=LEARNING_RATE_TUNE),loss=ls.BinaryCrossentropy(),metrics=["accuracy", "precision", "recall", "auc", "true_positives", "true_negatives", "false_positives", "false_negatives"]
)
history2=model.fit(train,validation_data=validation,epochs=50,callbacks=[callback])
model.save_weights("weights/model_with_data_augmentation.weights.h5")

with open("notebooks/first_fit.json","w") as f:
    json.dump(history1.history,f)

with open("notebooks/second_fit.json","w") as f:
    json.dump(history2.history,f)