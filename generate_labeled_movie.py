from pytorch.FasterRCNN.__main__ import predict_API, predict_one_API
import argparse
import os
import torch as t
from tqdm import tqdm
import cv2
from pytorch.FasterRCNN.models.faster_rcnn import FasterRCNNModel
import numpy as np
from pytorch.FasterRCNN.models import resnet

from pytorch.FasterRCNN.datasets import voc
from pytorch.FasterRCNN import state


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FasterRCNN")

    group = parser.add_mutually_exclusive_group()
    # group.add_argument("--generate-movie", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes and save movie")
    parser.add_argument("--output", metavar = "dir", action = "store", help = "save folder")
    parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
    group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
    parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")

    options = parser.parse_args()

    backbone = resnet.ResNetBackbone(architecture = resnet.Architecture.ResNet152)

    model = FasterRCNNModel(
        num_classes = voc.Dataset.num_classes,
        backbone = backbone,
        allow_edge_proposals = not options.exclude_edge_proposals
    ).cuda()

    if options.load_from:
        state.load(model = model, filepath = options.load_from)
    
    vs = cv2.VideoCapture(options.predict)
    writer = None

    while True:
        # grab the frame from the threaded video stream
        ret, frame = vs.read()
        if not ret:
            break
        
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        frame = predict_one_API(model = model, frame=frame, show_image = True, output_path = None)
        frame = np.array(frame)

        # Since PIL uses RGB and OpenCV uses BGR, we need to convert the color space
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img',frame)
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        
        if writer is None and options.output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(options.output, fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)
            
        # if the writer is not None, write the frame with recognized
        # faces to disk
        if writer is not None:
            writer.write(frame)

    cv2.destroyAllWindows()
    vs.release()

    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()