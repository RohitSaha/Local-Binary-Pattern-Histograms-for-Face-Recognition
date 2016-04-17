import numpy as np
import glob
import cv2


training_data = np.zeros((1, 62500))
labels = np.zeros((1, 4), 'float')
train = glob.glob('training_data/*.npz')
#extracting data from the saved .npz files
for i in train:
    with np.load(i) as data:
        print data.files
        training_temp = data['training_image_array']
        labels_temp = data['output_array']
    training_data = np.vstack((training_data, training_temp))
    labels = np.vstack((labels, labels_temp))


training_data = training_data[1:, :]
labels = labels[1:, :]

print training_data.shape
print labels.shape

e1 = cv2.getTickCount()

print "Learning.............."
model=cv2.createLBPHFaceRecognizer(radius=1,neighbors=9,grid_x=8,grid_y=8,threshold=120)
model.train(np.asarray(training_data),np.asarray(labels))

model.save("./trainer.xml")