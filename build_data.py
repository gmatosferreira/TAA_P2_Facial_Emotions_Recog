import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# Validate arguments
if len(sys.argv) == 1:
    print("Bad arguments! You must provide the base folder of the dataset.")
    print(f"Example: $ python {sys.argv[0]} data/train_set")
    exit()

basePath = sys.argv[1]

# Validate that folder has structure required 
if not os.path.exists(os.path.join(basePath, 'images')) or not os.path.exists(os.path.join(basePath, 'annotations')):
    print("Bad folder! The folder must have the following structure:")
    print("\t/images\t\t\tFor images")
    print("\t/annotations\t\tFor numpy files with annotations")
    exit()

# Ask user how many images they want to insert
processNumber = 0
while processNumber<1:
    print("How many images to you want to process?", end=" ")
    processNumber = int(input())

expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# For each image (sorted by name)
for f in sorted(os.listdir(basePath + '/images'), key=lambda x:int(x.split('.')[0])):
    # Validate that is file
    flocation = os.path.join(basePath + '/images', f)
    if not os.path.isfile(flocation):
        continue

    # If is, get annotations for file
    fid = f.split('.')[0]
    print(f'\nAnalysing {fid}')

    # Expression
    expression = np.load(os.path.join(basePath + '/annotations', fid + '_exp.npy'))
    expression = expressions[int(expression)]
    print('\tExpression', expression)

    # Arousal
    arousal = np.load(os.path.join(basePath + '/annotations', fid + '_aro.npy'))
    print('\tArousal', arousal)

    # Valence
    valence = np.load(os.path.join(basePath + '/annotations', fid + '_val.npy'))
    print('\tValence', valence)

    # Facial landmarks
    landmarks = np.load(os.path.join(basePath + '/annotations', fid + '_lnd.npy'))
    # print('Landmarks', landmarks)
    print('\tLandmarks.shape', landmarks.shape)

    # Process image
    img = Image.open(flocation)

    # Draw points on facial landmarks
    draw = ImageDraw.Draw(img)

    i = 0
    while i<landmarks.size:
        draw.point((landmarks[i], landmarks[i+1]), fill='yellow')
        i += 2

    # Save processed image (create folder if it does not exist)
    savedir = os.path.join(basePath, 'processed', expression)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    img.save(os.path.join(savedir, fid + '.jpg'))

    print('\tImage with facial landmarks dotted saved at', os.path.join(savedir, fid + '.jpg'))

    if (int(fid) >= processNumber):
        break