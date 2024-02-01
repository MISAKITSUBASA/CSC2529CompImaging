# Dynamic Recognition: Extending Faster RCNN Capabilities for Facial Identification in Media Stream

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```
Please note that because we are utilizing GPU for Training so you need to configure your device with [CUDA Computing Tool Kit](https://developer.nvidia.com/cuda-downloads) and [supproted cuDNN libraries](https://developer.nvidia.com/rdp/cudnn-download). Also You need to install [Opencv-python with CUDA support](https://www.youtube.com/watch?v=YsmhKar8oOc&t=450s)!

You need a CUDA support device to be able to run the code.
Please follow 

## Usage
```bash
# dataset side
# To modify the movie from 30FPS to 1 FPS for better sample collecting
python .\movie_modify.py -video PATH_OF_VIDEO --output PATH_OF_VIDEO_OUTPUT

# To save the face clip and corresponding bounding box coordinates.
python .\videro_test_V2.py -video PATH_OF_MODIFIED_VIDEO --output PATH_OF_FOLD_TO_SAVE 

# data set was splited and classified manually and put in to form like
_actors  (contains all face clips after manually classified)
___Courteney Cox
___David Schwimmer
...

and 

_DATA
__TRAINVAL
___Courteney Cox
___David Schwimmer
...
__TEST
___Courteney Cox
___David Schwimmer
...

# code to generate the XML file for annotation( but you need to change the base directory 'D:\\DSFD-Pytorch-Inference-1\\data\\actors' to "_actors directory"
python .\make_XML.py

# code to generate the Layour and Main text files for each person. (replace output_directory and base_directory as needed)
python .\generate_Layout.py  
python .\generate_Main.py  

# model side
# To Train the model, it will perform full training process.
python -m pytorch.FasterRCNN --train --dataset-dir=PATH_TO_DATA_SET --backbone=BACK_BONE --epochs=NUM_EPOCH --learning-rate=LEARNING_RATE --save-best-to=PAHT_TO_SAVE_MODEL

# To see the result from a single image:
python -m pytorch.FasterRCNN --backbone=BACKBONE--load-from=PATH_OF_THE_MODEL --predict=PATH_OF_IMAGE  

# To run the model and get a visual result from the movie
python .\generate_labeled_movie.py --output PATH_OF_OUTPUT_FILE_NAME  --predict PATHO_OF_MOIVE --load-from= PATH_OF_THE_MODEL    
```
IF you can not run the script you can watch the sample output we have.
## Limitation
limited to the Friends theories. 
## My Poster
[Our Poster](Poster.pdf)

## My Paper
[My Paper](Dynamic_Recognition__Extending_Faster_RCNN_Capabilities_for_Facial_Identification_in_Media_Stream.pdf)
