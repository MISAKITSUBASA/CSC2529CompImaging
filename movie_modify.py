import argparse
import cv2
import face_detection

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("--output", metavar = "dir", action = "store", help = "save folder")
args = vars(ap.parse_args())

# Initialize face detector
detector = face_detection.build_detector("DSFDDetector", max_resolution=1080)

# Initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["video"])
writer = None

# Variables to control frame processing rate
frame_count = 0
process_every_n_frames = 30

while True:
    # Read the next frame
    ret, frame = vs.read()

    # Check if the frame was read successfully
    if not ret:
        break
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    # Initialize video writer if it's not already done
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    # Process every nth frame
    if frame_count % process_every_n_frames == 0:
        if writer is not None:
            writer.write(frame)


    
    frame_count += 1

# Release the video writer and video stream
if writer is not None:
    writer.release()
vs.release()
