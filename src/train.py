from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.losses as ls
import tensorflow.keras.optimizers as op
import tensorflow.keras.callbacks as tfc
import Model

IMG_SIZE=(224,224)
BATCH_SIZE=32
LEARNING_RATE=0.0001

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
test=train=image_dataset_from_directory(
    "datasets/fanconic/skin-cancer-malignant-vs-benign/versions/4/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE


)
callback=tfc.EarlyStopping(patience=3,restore_best_weights=True)
Model.model.compile(optimizer=op.Adam(learning_rate=LEARNING_RATE),loss=ls.BinaryCrossentropy(),metrics=["accuracy"])
Model.model.fit(train,validation_data=validation,epochs=50,callbacks=[callback])
Model.model.save_weights("weights/model.weights.h5")