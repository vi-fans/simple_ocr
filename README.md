dataset source: http://hr-testcases.s3.amazonaws.com/2587/assets/sampleCaptchas.zip

recognise.py is a simple implementation using Histogram of Oriented Gradients (HOG) based recognition, basically capturing the contour information inside an image.

there is no deep learning involved, the main steps during the simple captcha recognition involve:
- detecting the contours after filtering the background noise according to the pixel values, each set of contour would be taken as a character for subsequent processing, (note: this part may fail if the characters are overlapping)
- for each character image provided in advance, compute the HOG based representation and store it, this would form the list of templates for comparison,
- next, given a query image, do the same to compute the HOG based representation, compare with the feature vector of each image in the list of templates, the character that corresponds to the nearest neighbour in the feature space is returned as the prediction.

the code can be executed with:
python3 recognise.py input/input100.jpg output100.txt

for reproducibility, the versions of the packages used are listed in requirements.txt

for comparison, recognise_tesseract.py utilises the existing ocr package in opencv, which is based on version 1.82.0 of leptonica.

references:
paper on HOG: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

