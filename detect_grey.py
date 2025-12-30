import cv2
import depthai as dai
import time
import numpy as np
import json

CAMERA_PREVIEW_DIM = (224, 224)
LABELS = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
YOLOV8N_CONFIG = "yolometalv3/best.json"
YOLOV8N_MODEL = "yolometalv3/best_openvino_2022.1_6shave.blob"
OUTPUT_VIDEO = "output_gray.mp4"


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def create_camera_pipeline(config_path, model_path):
    pipeline = dai.Pipeline()

    model_config = load_config(config_path)
    metadata = model_config.get("nn_config", {}).get("NN_specific_metadata", {})

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb_preview")

    detectionNetwork.setConfidenceThreshold(float(metadata.get("confidence_threshold", 0.5)))
    detectionNetwork.setNumClasses(metadata.get("classes", 6))
    detectionNetwork.setCoordinateSize(metadata.get("coordinates", 4))
    detectionNetwork.setAnchors(np.array(metadata.get("anchors", [])))
    detectionNetwork.setAnchorMasks(metadata.get("anchor_masks", {}))
    detectionNetwork.setIouThreshold(float(metadata.get("iou_threshold", 0.5)))
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.out.link(nnOut.input)
    detectionNetwork.passthrough.link(xoutRgb.input)

    return pipeline


pipeline = create_camera_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)

with dai.Device(pipeline) as device:
    detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)
    previewNN = device.getOutputQueue("rgb_preview", maxSize=4, blocking=False)

    fps = 30
    frame_width, frame_height = CAMERA_PREVIEW_DIM
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0

    while True:
        inPreview = previewNN.tryGet()
        if inPreview is not None:
            frame_rgb = inPreview.getCvFrame()
            frame_count += 1

            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            # Детекции ПЕРВЫМИ!
            inDet = detectionNN.tryGet()
            detections = []
            if inDet is not None:
                detections = inDet.detections
                print(f"Detections: {len(detections)}")

            for detection in detections:
                if detection.confidence > 0.6:
                    print(f"🎯 {LABELS[int(detection.label)]}: {detection.confidence:.2f}")

                    x1 = int(detection.xmin * frame_rgb.shape[1])
                    y1 = int(detection.ymin * frame_rgb.shape[0])
                    x2 = int(detection.xmax * frame_rgb.shape[1])
                    y2 = int(detection.ymax * frame_rgb.shape[0])

                    print(f"   PIXELS: ({x1},{y1})-({x2},{y2})")

                    # БЕЛЫЕ ТОЛСТЫЕ РАМКИ
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 5)
                    cv2.rectangle(frame_gray, (x1, y1), (x2, y2), 255, 3)

                    label = f"{LABELS[int(detection.label)]}: {detection.confidence:.2f}"
                    cv2.putText(frame_gray, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(frame_bgr)
            cv2.imshow("GRAY YOLO", frame_gray)


            if frame_count % 30 == 0:
                end_time = time.time()
                fps_actual = 30 / (end_time - start_time)
                print(f"FPS: {fps_actual:.2f}")
                start_time = end_time

        if cv2.waitKey(1) == ord('q'):
            break

out.release()
cv2.destroyAllWindows()
