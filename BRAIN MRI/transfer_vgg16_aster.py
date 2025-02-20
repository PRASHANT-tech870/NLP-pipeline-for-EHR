import tensorflow
from tensorflow import keras
from keras.layers import Dense,Flatten,Dropout,RandomFlip,RandomRotation
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)



)
conv_base.summary()
model1=Sequential()
model1.add(RandomFlip('horizontal'))
model1.add(RandomRotation(0.1))
model1.add(conv_base)
model1.add(Flatten())
model1.add(Dense(256,activation='relu',kernel_regularizer=keras.regularizers.l2(0.05)))
model1.add(Dropout(0.7))
model1.add(Dense(1,activation= 'sigmoid',kernel_regularizer= keras.regularizers.l2(0.05)))
model1.build(input_shape=(None,150,150,3))
model1.summary()
conv_base.trainable =False
train_ds = keras.utils.image_dataset_from_directory(
    directory ='/content/drive/MyDrive/Colab Notebooks/DATA_vgg16/Train',
    labels ='inferred',
    label_mode='int',
    batch_size=16,
    image_size=(150,150)

)
validation_ds =keras.utils.image_dataset_from_directory(
    directory='/content/drive/MyDrive/Colab Notebooks/DATA_vgg16/Validate',
    labels= 'inferred',
    label_mode ='int',
    batch_size= 16,
    image_size= (150,150)

)
model1.compile(optimizer =keras.optimizers.Adam(learning_rate=0.0001), loss= 'binary_crossentropy', metrics= ['accuracy'])
early_stopping =EarlyStopping(monitor='val_loss', patience=3)
history =model1.fit(train_ds, epochs=10, validation_data= validation_ds,callbacks=[early_stopping])
model1.save('/content/drive/MyDrive/Colab Notebooks/transfer_vgg16_model_aster.h5')
