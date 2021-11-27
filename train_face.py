from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from classifier import training

def train(data_dir = './aligned_img', model_path = './face_model/20180402-114759.pb', classifier_path = './class/classifier.pkl', debug=False):
    obj=training(data_dir,model_path,classifier_path)
    get_file=obj.main_train()
    if debug:
        print('Saved classifier model to file "%s"' % get_file)
    return 0
    
if __name__ == '__main__':
    train(debug=True)