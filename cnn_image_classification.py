# Convolutional Neural Network (CNN) is a type of neural network that is use to deal with image data,
# audio data or whenever you want to find a patterns in data.

# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt 
# from tensorflow.keras import datasets, layers, models
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# testing_images.shape
# # -- Dividing the pixel values of images by 255 is a common step in preprocessing images for training a machine learning model, 
# # -- particularly when working with image data represented in the RGB color space.
# # -- The pixel values in an RGB image range from 0 to 255, where 0 represents black and 255 represents 
# # -- full intensity for the respective color channels (red, green, blue). However, most machine learning algorithms work more effectively 
# # -- with input data that is normalized or scaled between 0 and 1.
# # -- By dividing the pixel values by 255, you are effectively scaling the values down to the range of 0 to 1. 
# # -- This normalization ensures that the image data falls within a consistent range, making it easier for the model to learn 
# # -- patterns and generalize from the training data to unseen images.
# # -- Additionally, normalizing the pixel values can also help prevent certain issues during optimization. 
# # -- Some optimization algorithms can be sensitive to the scale of the input features, and normalizing the pixel values helps mitigate this.
# # -- In the given code snippet, it appears that the images are being divided by 255 to normalize 
# # -- the pixel values before using them for training and testing a machine learning model.

# training_images, testing_images = training_images/ 255, testing_images/ 255

# class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# for i in range(16):
#     plt.subplot(4,4,i+1) # this means that we have 4x4 columns and each iterate will +1. i=0 then in here will be 1 so it will place in the first column
#     plt.xticks([]) # give empty list is because we don't want coordinate so it will not show out.
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary) # cmap means binary colour map. ** But why use this colour map?
#     plt.xlabel(class_names[training_labels[i][0]])

# plt.show()

# # training_images = training_images[:20000]
# # training_labels = training_labels[:20000] # these 2 codes are optional, reduced it so the running time can be faster.

# # testing_images = testing_images[:40000]
# # testing_labels = testing_labels[:40000]

# # -- Build Model
# model = models.Sequential()
# # Conv2D layers filter out the feature of the images for example horse has long legs, cat has point ears etc
# model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (32,32,3))) # input shape is 32x32 pixels and 3 colour channels which are RGB
# # Maxpooling2D reduces the image to the essential information
# model.add(layers.MaxPooling2D(2,2)) 
# # Then use a convolutional layer again to process the result
# model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

# model.add(layers.Flatten())

# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax')) 
# # -- using softmax here as output is because it can scales all the result and add up to one so if you have 50, 60 or 70 then it will
# # -- just scale them so that they are percentages and adding up to one so that we will get a distribution of probabilities.
# # -- get percentage or probablities for the classifications.

# # -- After specify all the layers then we have to compile
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # -- Feed the model with training datasets
# # -- epochs basically means how often is the model going to the same data.
# # -- if we state 10 epochs meaning that the model is going to see the same data 10 times
# model.fit(training_images, training_labels, epochs=10, validation_data = (testing_images, testing_labels)) 

# loss, accuracy = model.evaluate(testing_images, testing_labels)

# print(f'Loss: {loss}')
# print(f'Accuracy" {accuracy}') 
# # -- Accuracy is ~65%, it is consider quite low but there are 10 classifications where neural network dont know what they are, so this 66% consider pretty decent

# # this is to save the model so we dont have to keep train all over again when ever we run
# # -- the model has been loaded in the same directory so we can not command it from line 30 to 37 and 45 to 81
# model.save('image_classifier.model')



## --------------------------------------------------------- ##
# -- after run the code lines above, we can all command it (this step is 1st step)
# -- we saved the model already hence all the code lines above are actually not needed anymore.
# -- we just need to call our model name as image_classifier.model to rune predictions (this is 2nd step, below code lines)


model = models.load_model('image_classifier.model')

# -- the following step is very important because when we are using imread function, we load the picture with BGR instead of RGB
# -- but till now the whole project is using RGB, which means the model is trained using RGB images as well as using matplotlib in visualization also RGB
img = cv.imread('/Users/johnwong/Desktop/VS Code Scripts/Personal Side Projects/Image_Classifications/image_folder/deer.jpg')

# -- so in the following step, we have to swap the colour scheme from BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# -- we trained our model to accept image with the shape of 32,32
# -- but this picture is not so will run to error, hence this code will help us to reshape it.
img = cv.resize(img, (32, 32))

plt.imshow(img, cmap=plt.cm.binary)

# -- here need to pass in numpy because our model has certain structure
# -- need to scale all input
# -- this prediction is we get 10 activations of the 10 softmax neruons, so we want to get the maximum value of the predictions
prediction = model.predict(np.array([img]) / 255)

# -- what argmax do is give us the idex of the maximum value from the neurons
index = np.argmax(prediction)

print(f'Prediction is {class_names[index]}')

plt.show()