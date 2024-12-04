# cnn-cats-dogs-classifier
This project uses a custom Convolutional Neural Network (CNN) to classify images of cats and dogs in their respective classes.

This was the first computer vision project I made after learning about CNNs from prof. Andrew Ng's course on
computer vision. My aim for this project was to learn practical implementation of CNNs
as a classifier.

I used a dataset containing many thousands of images of cats and dogs. I selected 4000 of these images as my training set, and 200 each for 
validation and testing sets. A custom CNN model was created using Keras. I tested various models for the same and chose the one which
gave the best accuracy on the validation set after training for a few epochs.

Overfitting was a major problem I faced during training where the model would perform really well on training data but
did not give good accuracy on validation data. I used Dropout in a few places to overcome this problem to some
extent. I realized that my custom models were not very complex and even though I had used good standard practices 
to design my model, it fell short of predefined very large models like VGG16 when it came to accuracy. But, my aim 
was not to get the best accuracy on any model, I wanted to create a 'CUSTOM' model which could perform relatively well 
on this dataset.

At the end, I was able to get a training accuracy of around 75% and a testing accuracy of 73.75%. Accuracy can be used
as a good metric since the number of images for dogs and cats were equal.
