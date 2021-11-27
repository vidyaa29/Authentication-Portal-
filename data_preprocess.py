from preprocess import preprocesses

def data_preprocess(input_dir = './train_img', output_dir = './aligned_img', debug=False):
    obj=preprocesses(input_dir,output_dir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()
    if debug:
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    return (nrof_images_total, nrof_successfully_aligned)

if __name__=="__main__":
    data_preprocess()