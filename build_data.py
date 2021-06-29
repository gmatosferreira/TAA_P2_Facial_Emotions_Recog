import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# To clear
# rm -r data/train_set/processed/

# Validate arguments
if len(sys.argv) < 3:
    print("Bad arguments! You must provide the base folder of the dataset.")
    print(f"Example: $ python {sys.argv[0]} data/train_set <numberOfImagesToProcess>")
    exit()

basePath = sys.argv[1]
processNumber = int(sys.argv[2])

# Validate that folder has structure required 
if not os.path.exists(os.path.join(basePath, 'images')) or not os.path.exists(os.path.join(basePath, 'annotations')):
    print("Bad folder! The folder must have the following structure:")
    print("\t/images\t\t\tFor images")
    print("\t/annotations\t\tFor numpy files with annotations")
    exit() 
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

    # Crop facial landmarks
    markerN = 1
    leftEye = []
    rightEye = []
    mouth = []
    face = []
    nose = []
    i = 0
    while i<landmarks.size:
        coords = (landmarks[i], landmarks[i+1])
        # Distinguish different elements
        # https://www.researchgate.net/figure/The-ibug-68-facial-landmark-points-mark-up_fig9_327500528
        if markerN<=27: # Face limit
            face.append(coords)
        elif markerN<=36: # Nose
            fill = 'hotpink'
            nose.append(coords)
        elif markerN<=42: # Left eye
            fill = 'red'
            leftEye.append(coords)
        elif markerN<=48: # Right eye
            fill = 'orange'
            rightEye.append(coords)
        else: # Until 68, mouth
            fill = 'purple'
            mouth.append(coords)
        i += 2
        markerN += 1
    
    print("\tface", (max(min([x[0] for x in face]), 0), max(0, min([x[1] for x in face])), min(224, max([x[0] for x in face])), min(max([x[1] for x in face]), 224)))

    # im.crop((left, top, right, bottom))
    leftEyeImg = img.crop((min([x[0] for x in leftEye]), min([x[1] for x in leftEye]), max([x[0] for x in leftEye]), max([x[1] for x in leftEye])))
    rightEyeImg = img.crop((min([x[0] for x in rightEye]), min([x[1] for x in rightEye]), max([x[0] for x in rightEye]), max([x[1] for x in rightEye])))
    mouthImg = img.crop((min([x[0] for x in mouth]), min([x[1] for x in mouth]), max([x[0] for x in mouth]), max([x[1] for x in mouth])))
    faceImg = img.crop((max(min([x[0] for x in face]), 0), max(0, min([x[1] for x in face])), min(224, max([x[0] for x in face])), min(max([x[1] for x in face]), 224)))
    noseImg = img.crop((min([x[0] for x in nose]), min([x[1] for x in nose]), max([x[0] for x in nose]), max([x[1] for x in nose])))

    # Save processed images (create folder if it does not exist)
    savedir = os.path.join(basePath, 'processed', expression)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Resize face imgs
    faceImg = faceImg.resize((150,150))
    
    # leftEyeImg.save(os.path.join(savedir, fid + '_lefteye.jpg'))
    # rightEyeImg.save(os.path.join(savedir, fid + '_righteye.jpg'))
    # mouthImg.save(os.path.join(savedir, fid + '_mouth.jpg'))
    faceImg.save(os.path.join(savedir, fid + '_face.jpg'))
    # noseImg.save(os.path.join(savedir, fid + '_noseeye.jpg'))

    # Point facial landmarks
    i = 0
    draw = ImageDraw.Draw(img)
    while i<landmarks.size:
        draw.point((landmarks[i], landmarks[i+1]), fill='white')
        i += 2

    img.save(os.path.join(savedir, fid + '.jpg'))


    print('\tImage with facial landmarks dotted saved at', os.path.join(savedir, fid + '(_mouth/_nose/_leftEye/_rightEye/_face).jpg'))

    if (int(fid) >= processNumber):
        break