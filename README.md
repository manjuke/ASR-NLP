# ASR-ASSIGNMENT
Objective of the problem is given an “audio” file, we need to do ASR & NLP and derive it’s transcription, and
corresponding action, object, and location.
This assignment has 2 problems to be solved:
1. ASR to derive the text from the given audio
2. NLP to find intent of the transcribed text w.r.to action, object, and location fields

PART I : ASR:
I have used the pre-trained models of Nvidia Nemo and decoded the wave files of validation set using 3 Nvidia
Nemo models, namely – Quartznet, Jasper, Conformer-CTC-large. These models are built on top of pytorch &
pytorch lighting, but does not GPU for decoding.
The WER’s obtained are as below:
Quartznet – 50.412%
Jasper - 47.805%
Conformer-CTC-large - 40.456 %


Part II : NLP for intent classification
The SVM classifers are trained to classify the intent of action, object, & location fields. The word
embeddings for the transcriptions of training data are generated using Spacy NLP model. Totally,
three SVM classifiers are trained, one for each action, object, & location categories. The
corresponding values in action, object, & location fields are used as labels for training. The
validation tests are used for SVM validation and testing. The accuracy of SVMs predicting action,
location, and object are found to be: 80.69% 88.04% 79.37% respectively.
It is found that all the three SVM’s have shown a accuracy of 100% on validation sets using correct
transcriptions.
