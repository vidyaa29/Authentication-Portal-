from data_utils import *
from model import load_model
import config
import predict_face
import tensorflow as tf

graph = tf.get_default_graph()

class Inference():
    def __init__(self):
        self.voice_model = load_model()

    def preprocess_audio(self,filepath):
        aud, sr = librosa.load(filepath)
        print(aud.shape)
        aud = remove_silence(aud)

        mfcc,chroma,mel,contrast,tonnetz = extract_feat(aud,sr)
        data = np.concatenate((mfcc,chroma,mel,contrast,tonnetz))
        return data

    def predictVoice(self,voice_path):
        with graph.as_default():
            voice_model1 = load_model()
            data     = self.preprocess_audio(voice_path)

            # print(voice_model1.summary())
            prediction = voice_model1.predict(np.array([[data]]))
            # print("Prediction: ",prediction[0])

            pred_class = np.array(prediction).argmax()
            class_to_name = load_json_data(config.VOICE_CLASSES)
            empID = class_to_name[str(pred_class)]

            if empID==0000:
                return "Unknown"
            # return str(empID)
            data = fetchDataFromDB(empID)[0]
            return {"EmpID":data[0], "FullName":data[1], "EmailID":data[2]}

    def predictFace(self,image_path):
        className = predict_face.predict(frame_path=image_path)
        return className