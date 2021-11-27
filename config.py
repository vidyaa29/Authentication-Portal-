import os
import glob


dir_this = os.path.dirname(__file__)

INCOMING_VIDEO_PATH = os.path.join(dir_this, "recorded_video\\uploaded_video.mp4")
PREDICT_IMAGE_PATH  = os.path.join(dir_this, "face_image_to_predict\\predict_image.jpg")

DIR_IN_FILE = os.path.join(dir_this, ".\\Voice_Training_Samples\\")
TRAINED_DIR = os.path.join(dir_this, ".\\Voice_Trained_Samples\\")

VOICE_CLASSES  = os.path.join(dir_this, "voice_features\\classNames.json")
VOICE_FEATURES = os.path.join(dir_this, "voice_features\\TrainSet.pkl")

MODEL_PATH          = os.path.join(dir_this,"voice_model\\model.json")
MODEL_WEIGHTS_PATH  = os.path.join(dir_this,"voice_model\\model_weights.h5")

SQLITE_DB_PATH = os.path.join(dir_this,"database\\EMP_Details.db")

TRAIN_IMAGES_PATH = os.path.join(dir_this,"train_img")