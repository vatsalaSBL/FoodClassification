function predictFoodClass()

% load the network and classifier parameters from the saved files
net = load('net.mat');
classifier = load('classifier.mat');


% put the test images in the tst directory, single image tst can also be
% done
dnflder = './tst';
 
%store test set db
imds = imageDatastore(dnflder, 'IncludeSubfolders',true);

imageSize = net.net.Layers(1).InputSize;
%Preprocess
augmentedTestSet = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');
featureLayer = 'fc1000';

% Extract test features using the CNN
testFeatures = activations(net.net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');


% Pass CNN image features to trained classifier
predictedLabels = predict(classifier.classifier, testFeatures, 'ObservationsIn', 'columns')


