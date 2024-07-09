import python3 as py;
string test(string img_string):=embed(py)

import tensorflow
import base64
from tensorflow import keras
from keras.models import load_model
import numpy as np
import io
from PIL import Image

model =load_model('/var/lib/HPCCSystems/mydropzone/new_vgg_tumor_transfer.h5')


byte=img_string


image_data = base64.b64decode(img_string)

image = Image.open(io.BytesIO(image_data))

image = image.resize((224, 224))

image = image.convert('RGB')

image=np.array(image)
image=np.reshape(image,(1,224,224,3))


predictions =model.predict(image)
predicted_label =np.argmax(predictions, axis=1)
predicted_prob =np.max(predictions,axis=1)


class_labels=['glioma','meningioma','notumor',pituitary]
predicted_class =class_labels[predicted_label[0]]
predicted_prob1 =predicted_prob[0]


return f"{predicted_class},{predicted_prob1:.2f}"

endembed;

export mri_roxie22() := FUNCTION
    rec:=RECORD
        STRING STR;
    END;

    STRING img_string := 'none' : STORED('img_string');
  
    op := test(img_string);
    RETURN op;

  
end;
