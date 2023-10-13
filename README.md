# YOLO-bbox2seg
Converts a YOLO format bounding box dataset to a segmentation version using Meta's SAM (Segment Anything Network). This is a modification of the automatic annotation feature provided by ultralytics, but instead uses a pre-existing dataset to generate the masks instead of using trained weights. 

This will output the YOLO-format segmentation labels in ``/seg_labels`` and **optionally** images with the mask outlines in ``seg_images`` for sanity checking.

## Installation
This requires the ultralytics pip package, so you can just follow the installation instructions from the [Ultralytics Repository](https://github.com/ultralytics/ultralytics)

## Usage
``python bbox2seg.py YAML_PATH IMAGE_DIRECTORY LABEL_DIRECTORY SEG_IMAGES(y/n) USE_LARGE(y/n)``

* ``YAML_PATH``: Path to ``data.yaml`` in dataset
* ``IMAGE_DIRECTORY``: Directory of images
* ``LABEL_DIRECTORY``: Directory of corresponding labels in yolo bounding-box format
    * i.e ``images/train1.jpg`` has the corresponding label ``label/train1.txt``
* ``SEG_IMAGES``: Output copy of dataset images with generated mask applied (y/n)
* ``USE_LARGE``: Use SAM large instead of SAM base (y/n)