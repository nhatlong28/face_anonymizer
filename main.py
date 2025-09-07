import cv2
import os
import argparse
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
def processing_img(img, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    face_detector_result = detector.detect(mp_image)

    #if face_detector_result.detections:
    for face in face_detector_result.detections:
        bbox = face.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        # Anonymize face in the bounding box
        face_region = img[y:y + h, x:x + w]
        face_region = cv2.GaussianBlur(face_region, (51, 51), 20)
        img[y:y + h, x:x + w] = face_region
    return img

args = argparse.ArgumentParser()
args.add_argument('--mode', default='image')
args.add_argument('--file_path', default=os.path.join('.', 'data', 'human_face.jpg'))
args = args.parse_args()

# Direct to store output
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='./blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5)
with FaceDetector.create_from_options(options) as detector:
    if args.mode == "image":
        img = cv2.imread(args.file_path)
        img = processing_img(img, detector)
        cv2.imwrite(os.path.join(output_dir, 'output.jpg'), img)
        cv2.waitKey(0)

    elif args.mode == "webcam":    
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = processing_img(frame, detector)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.file_path)
        ret, frame = cap.read()
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
        while ret: 
            frame = processing_img(frame, detector)
            output_video.write(frame)
            ret, frame = cap.read()
        
        cap.release()
        output_video.release()


    