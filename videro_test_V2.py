import argparse
import cv2
import face_detection
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to output video")
args = vars(ap.parse_args())

# Initialize the face detector
detector = face_detection.build_detector("DSFDDetector", max_resolution=1080)

# Initialize the video stream and writer
vs = cv2.VideoCapture(args["video"])
writer = None

# Directory for saving training images
save_dir = "training_images"
os.makedirs(save_dir, exist_ok=True)
img_counter = 0

frame_counter = 0

def save_face_clips(frame, bboxes):
    global img_counter
    global frame_counter
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        # Crop and save face image
        frame_copy = frame.copy()
        try:
            x0 = x0-20
            y0 = y0-20
            x1 = x1+20
            y1 = y1+20
            face_img = frame_copy[y0:y1, x0:x1]
            img_path = os.path.join(save_dir, f"face_{img_counter}_frame_{frame_counter}@3_x0_{x0}_y0_{y0}_x1_{x1}_y1_{y1}.jpg")
            cv2.imwrite(img_path, face_img)
            img_counter += 1
            # draw rectangle on face
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        except:
            pass
        

# Process video frames
while True:
    
    frame_counter += 1
    
    ret, frame = vs.read()
    if not ret:
        break

    dets = detector.detect(frame[:, :, ::-1])[:, :4]
    save_face_clips(frame, dets)

    # cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    
    if writer is not None:
        writer.write(frame)

cv2.destroyAllWindows()
vs.release()
if writer is not None:
    writer.release()
