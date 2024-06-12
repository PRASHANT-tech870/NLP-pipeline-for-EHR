import streamlit as st

import requests

from PIL import Image

import base64

url="http://localhost:8002/WsEcl/json/query/roxie/mri_roxie_9.1"

st.title("classification mri")


uploaded_file=st.file_uploader("upload an image",type=["png","jpg","jpeg"])

if st.button("predict"):

    if uploaded_file:
        img=Image.open(uploaded_file)
        img_str=base64.b64encode(img.tobytes())
        st.write(img_str)

        data={
            "mri_roxie_9.1":{
                "img_string": str(img_str.decode('utf-8'))

            }
        }

        response=requests.post(url,json=data)

        if response.status_code==200:
            json_response=response.json()

            predicted_label=json_response


            st.write("predicted_label",predicted_label)

        else:
            st.error("error,request failed with status code {}".format(response.status_code))

    else:
        st.error("error , no image uploaded")
