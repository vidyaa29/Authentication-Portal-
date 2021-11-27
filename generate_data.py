from tqdm import tqdm
import time
import pandas as pd, numpy as np
import glob
import random
import config

from data_utils import *

# base_path   = "./Training_Samples/"

# audio_df = pd.read_pickle("TrainSet.pkl")

def data_generator():
    augmentations = {"speed":speed_aug,"volume":volume_aug,"noise":noise_aug}
    class_names = load_json_data(config.VOICE_CLASSES)
    audio_df = pd.read_pickle(config.VOICE_FEATURES)

    for eid,class_name in enumerate(os.listdir(config.DIR_IN_FILE)):
        begin = time.time()
        class_names[str(eid)] = class_name
        for filename in tqdm(glob.glob(f"{config.DIR_IN_FILE}/{class_name}/*.wav")):
            n_samples = np.random.randint(1,12)
            for i in range(n_samples):
                aud, sr = librosa.load(filename)
                aud = remove_silence(aud)

                augs = list(augmentations.keys())
                random.shuffle(augs)
                
                # augmentation is happening here
                aud = augmentations[augs[0]](aud)
                mfcc,chroma,mel,contrast,tonnetz = extract_feat(aud,sr)
                data = np.concatenate((np.array([class_name]),mfcc,chroma,mel,contrast,tonnetz))
                audio_df.loc[len(audio_df)] = data
        print(f"Took {time.time()-begin} seconds for {class_name}")

    write_json_data(config.VOICE_CLASSES,class_names)
    audio_df.to_pickle(config.VOICE_FEATURES)
    move_files(config.DIR_IN_FILE,config.TRAINED_DIR)

if __name__ == '__main__':
    data_generator()