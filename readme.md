** How Old Are You?
*** Convolutional Neural Net solves the problem of Human Age Prediction

wide_resnet.py file and some example code borrowed from:
- https://github.com/yu4u/age-gender-estimation

This program expects Python3 (please consider using a vitualenv)

To install required packages please run

```pip install -r requirements.txt```

Predict age on images from default images dir (./images) using pretrained model from repo:

```python how_old_are_you.py```

Predict age on images from user specified images dir using pretrained model:

```python how_old_are_you.py -i [path_to_images]```

Predict age on images from images dir using specified weights file:

```python how_old_are_you.py -w [path_to_weights_file]```

Train new network (assumes either UTKFace training dataset is in ./training/ dir or db file "utk.mat" is in current dir):

```python how_old_are_you.py -t```
