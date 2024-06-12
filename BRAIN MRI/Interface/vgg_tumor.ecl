import python3 as py;
integer test1()  := embed(py)
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout,RandomFlip,RandomRotation
from keras.callbacks import EarlyStopping
import numpy as np
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)



)

model1=Sequential()
model1.add(RandomFlip('horizontal'))
model1.add(RandomRotation(0.1))
model1.add(conv_base)
model1.add(Flatten())
model1.add(Dense(256,activation='relu',kernel_regularizer=keras.regularizers.l2(0.05)))
model1.add(Dropout(0.7))
model1.add(Dense(4,activation= 'softmax',kernel_regularizer= keras.regularizers.l2(0.05)))



model1.build(input_shape=(None,224,224,3))

conv_base.trainable =False

train_ds = keras.utils.image_dataset_from_directory(
    directory ='/var/lib/HPCCSystems/mydropzone/tumor_dataset/Archive_tumor_kaggle/Training',
    labels ='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224,224)

)
validation_ds =keras.utils.image_dataset_from_directory(
    directory='/var/lib/HPCCSystems/mydropzone/tumor_dataset/Archive_tumor_kaggle/Validation',
    labels= 'inferred',
    label_mode ='categorical',
    batch_size= 16,
    image_size= (224,224)

)

model1.compile(optimizer =keras.optimizers.Adam(learning_rate=0.0001), loss= 'binary_crossentropy', metrics= ['accuracy'])

early_stopping =EarlyStopping(monitor='val_loss', patience=3)

history =model1.fit(train_ds, epochs=10, validation_data= validation_ds,callbacks=[early_stopping])
model1.save('/var/lib/HPCCSystems/mydropzone/new_vgg_tumor_transfer.h5')



return 1
endembed;
test1();
