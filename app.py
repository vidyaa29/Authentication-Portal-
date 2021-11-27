import json
from flask import Flask, jsonify, abort, request,  Response
from flask_cors import CORS
import time
import os
import config
import uuid
from data_utils import addDataToDB, save_frames_from_video
import generate_data, train
from inference import Inference
from flask_uploads import UploadSet
from werkzeug.utils import secure_filename
import train_face, data_preprocess
from data_utils import *



app = Flask(__name__)
cors = CORS(app)
infer = Inference()

def get_time_string():
    return time.strftime("%d-%m-%Y-T-%H-%M-%S")

def save_audio(request, audio_path,param='audio'):
    session_id = str(uuid.uuid4())
    aud_file = request.files[param]
    print(aud_file)
    t = get_time_string()
    fpath_aud = os.path.join(audio_path, "{}-{}.wav".format(session_id, t))
    aud_file.save(fpath_aud)
    return fpath_aud

def save_image(request, image_path,param='image'):
    img_file = request.files[param]
    print(img_file)
    img_file.save(image_path)
    return image_path

@app.route('/addAudioFile', methods=['POST'])
def addAudioFile():
    if request.form.get('label', True):
        label = request.args.get('label')

    if not os.path.exists(f"{config.TRAINED_DIR}/{label}"):
        new_path = os.path.join(config.DIR_IN_FILE,label)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
    else:
        return {"Error": "Emp ID already exists... Try Again !!"}
    try:
        _ = save_audio(request,new_path)
    except Exception as e:
        status_code = Response(status=201,response=e)
        return status_code

    return "Thank You"

@app.route('/addEmpDetails', methods=['POST'])
def addEmpDetails():
    if request.form.get('empId', True):
        empId = request.args.get('empId')
    if request.form.get('fullName', True):
        fullName = request.args.get('fullName')
    if request.form.get('email', True):
        emailID = request.args.get('email')
    try:
        addDataToDB(empId,fullName,emailID)
    except Exception as e:
        print(e)
        return {"Error": str(e)}
    
    return "Added succesfully huha"

@app.route('/trainVoice', methods=['GET'])
def trainVoice():
    try:
        generate_data.data_generator()
        # train.train()
        return "Model has been trained succesfully"
    except Exception as e:
        return {"Error": "Some internal error has occurred"}

@app.route('/predictVoice', methods=['POST'])
def predictVoice():
    try:
        audio_path = save_audio(request,"./predicted_voices")
        empData = infer.predictVoice(audio_path)
        return json.dumps(empData)
    except Exception as e:
        print(e)
        return {"Error": "Some internal server error has occurred"}

@app.route('/uploadVideo', methods=['POST'])
def uploadVideo():
    if request.form.get('label', True):
        label = request.args.get('label')
    else:
        return {"Error": "Please pass the 'label' parameter"}

    if not os.path.exists(f"{config.TRAIN_IMAGES_PATH}/{label}"):
        new_path = os.path.join(config.TRAIN_IMAGES_PATH,label)
        os.makedirs(new_path)
    else:
        return {"Error": "Emp ID already exists... Try Again !!"}

    # media = UploadSet('media', ('mp4')) # Create an upload set that only allow mp4 file
    if "video" in request.files:
        video = request.files["video"]
        print(video)
        video.save(config.INCOMING_VIDEO_PATH)
        save_frames_from_video(config.INCOMING_VIDEO_PATH,frames_path=new_path)

    else:
        return {"Error": "Please pass the 'video' parameter"}
    
    return "Video save ho gya"
    # Video saved


@app.route('/trainFace', methods=['GET'])
def trainFace():
    try:
        data_preprocess.data_preprocess()
        train_face.train(debug=True)
    except Exception as e:
        return {"Error": "Some internal error has occurred"}
    return "Model Trained Succesfully"

@app.route('/predictFace', methods=['POST'])
def predictFace():
    try:
        img_path = save_image(request,config.PREDICT_IMAGE_PATH)
        resp_class = infer.predictFace(img_path)
        print(resp_class)

        # empID = resp_class[0]
        # if empID==0000:
        #         return "Unknown"
        # # return str(empID)
        # data = fetchDataFromDB(empID)[0]
        # return {"EmpID":data[0], "FullName":data[1], "EmailID":data[2]}

        return str(resp_class)
    except Exception as e:
        return {"Error": "Some internal error has occurred"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5699)