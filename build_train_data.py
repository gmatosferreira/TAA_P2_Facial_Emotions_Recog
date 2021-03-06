import numpy as np
import scipy.io as scp
import os 
from PIL import Image 
import cv2 as cv

subsets = {
    'training': {
        'dir': './data/train_set/processed/',
        'start': 0,
        'samples': 2400,
        'folder_crop': './datasets/crop/train',
        'folder_no_crop': './datasets/no_crop/train'
    },
    'test': {
        'dir': './data/val_set/processed/',
        'start': 0,
        'samples': 400,
        'folder_crop': './datasets/crop/test',
        'folder_no_crop': './datasets/no_crop/test'
    }
}

emotionsSet = [['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], ['Fear', 'Anger', 'Surprise','Neutral'], ['Happy', 'Anger', 'Neutral'], ['Fear', 'Anger'], ['Happy', 'Sad'], ['Happy', 'Neutral'], ['Fear', 'Anger', 'Surprise'], ['Fear', 'Sad', 'Happy'] ]



# Create folders if they dont exist
for s in subsets.values():
    if not os.path.exists(s['folder_crop']):
        os.makedirs(s['folder_crop'])
        for e in emotionsSet[0]:
            os.makedirs(s['folder_crop']+'/'+e)
    if not os.path.exists(s['folder_no_crop']):
        os.makedirs(s['folder_no_crop'])
        for e in emotionsSet[0]:
            os.makedirs(s['folder_no_crop']+'/'+e)


# Foreach emotion set
for emotions in emotionsSet:
    print("Emotion set",emotions)

    # Foreach subset 
    for subset, subsetinfo in subsets.items():
        data_dic_crop ={"X":[],"y":[]}
        data_dic_no_crop ={"X":[],"y":[]}

        print("Subset", subset)
        i = 0

        # Foreach emotion
        for e in emotions:
            dir = subsetinfo['dir'] + e + '/'
            entries = sorted(os.listdir(dir), key=lambda x:int(x.split('.')[0].split('_')[0]))
            cnt = 1
            print(i, e, end="\t\t\t")

            for image in entries[subsetinfo['start']::]:
                if cnt>subsetinfo['samples']:
                    break
                
                if "face" in image:
                    #croped
                    image = Image.open(dir + image).convert('L')
                    image = np.array(image)
                    image= cv.equalizeHist(image)
                    image = image/255
                    data_dic_crop["X"].append(image)
                    data_dic_crop["y"].append([i])
                    cnt+=1

                else:
                    #not cropped
                    image = Image.open(dir + image).convert('L')
                    image = np.array(image)
                    image= cv.equalizeHist(image)
                    image = image/255
                    data_dic_no_crop["X"].append(image)
                    data_dic_no_crop["y"].append([i])
                    cnt+=1
            
            print(f"Got {cnt-1} images!")
            i += 1    
        
        filename_crop =  subsetinfo['folder_crop'] +'/' + '_'.join(emotions) + '.mat'
        filename_no_crop =  subsetinfo['folder_no_crop'] +'/' + '_'.join(emotions) + '.mat'
        scp.savemat(filename_crop, data_dic_crop)
        scp.savemat(filename_no_crop, data_dic_no_crop)
        
        print("Croped data:")
        print(f"Saved at {filename_crop}\n")
        print("X", len(data_dic_crop["X"]))
        print("X[0].shape", data_dic_crop["X"][0].shape)
        print("y", len(data_dic_crop["y"]))
        print("\n\n")

        print("No Croped data:")
        print(f"Saved at {filename_no_crop}\n")
        print("X", len(data_dic_no_crop["X"]))
        print("X[0].shape", data_dic_no_crop["X"][0].shape)
        print("y", len(data_dic_no_crop["y"]))
        print("\n\n")
        
    print()
    

