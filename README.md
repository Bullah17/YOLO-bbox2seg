# YOLO-bbox2seg
Converts a YOLO format bounding box dataset to a segmentation version using Meta's SAM (Segment Anything Network). This is a modification of the automatic annotation feature provided by ultralytics, but instead uses a pre-existing dataset to generate the masks instead of using trained weights. 


This will output the YOLO-format segmentation labels in ``/seg_labels`` and images with the mask outlines in ``seg_images`` for sanity checking.

## Warning
This is currently configured to work with maximum 6 classes in a dataset. This is due to the color allocation on the line after the function definition for ``visualise_segment()``. Feel free to add your own rgb values to this dictionary if you need.

## Installation
This requires the ultralytics pip package, so you can just follow the installation instructions from the [Ultralytics Repository](https://github.com/ultralytics/ultralytics)

## Usage
``python bbox2seg.py IMAGE_DIRECTORY LABEL_DIRECTORY``

* ``IMAGE_DIRECTORY``: Directory of images
* ``LABEL_DIRECTORY``: Directory of corresponding labels in yolo bounding-box format
* i.e ``images/train1.jpg`` has the corresponding label ``label/train1.txt``
