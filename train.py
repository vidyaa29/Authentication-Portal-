from data_utils import load_json_data
import pandas as pd
from model import *
import config

def train():
    classes = load_json_data(config.VOICE_CLASSES)
    print(classes)

    model = get_model(n_classes=len(classes.keys()))

    opt = Adamax(lr = 1e-3, decay = 1e-5) # Adamax has shown to yield faster learning than Adam and SGD
    model.compile(loss = 'categorical_crossentropy', 
                optimizer = opt,
                metrics = ['accuracy'])
    
    # Add automated stopping after val_loss difference from epoch t and t-1 is 
    # more than 0.001; give it three more epochs to try and get back on track (patience)
    earlystop = EarlyStopping(monitor='val_loss',
                                min_delta=0.001,
                                patience=3,
                                verbose=0, mode='auto')
    
    audio_df = pd.read_pickle(config.VOICE_FEATURES)
    y = audio_df.id
    x = audio_df.iloc[:,1:]    
    # Train, test, split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 121)

    # TRAIN SET
    x_train_array = x_train.values    # convert df to array
    x_train = np.reshape(x_train_array, newshape = (len(x_train), 1,192))

    y_train_cat = pd.get_dummies(y_train) # converts variables in one column to have their own column
    y_train = y_train_cat.values 

    # VALIDATION SET
    x_test_array = x_test.values      # convert df to array
    x_val = np.reshape(x_test_array, newshape = (len(x_test), 1, 192))

    y_test_cat = pd.get_dummies(y_test) # converts variables in one column to have their own column
    y_val = y_test_cat.values
    print(x_val.shape,y_val.shape)

    # fit the data and save it with history variable. 
    history = model.fit(x_train, 
                        y_train, 
                        epochs = 100, 
                        batch_size = 25,
                        validation_data= (x_val,y_val), callbacks = [earlystop])
    
    # Save best model so don't have to rerun in scenario of crash
    model_json = model.to_json()
    with open(config.MODEL_PATH, "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(config.MODEL_WEIGHTS_PATH)
    print("Saved Model")
    
if __name__ == '__main__':
    train()