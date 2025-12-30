import cv2
import depthai as dai
import time
import numpy as np


# Константы
CAMERA_PREVIEW_DIM = (224, 224)
LABELS = [ "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches"
           ]
YOLOV8N_CONFIG = "yolometalv3/best.json"  # Укажите путь
YOLOV8N_MODEL = "yolometalv3/best_openvino_2022.1_6shave.blob"  # Укажите путь
OUTPUT_VIDEO = "output.mp4"  # Путь для сохранения видео


def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        return json.load(f)


def create_camera_pipeline(config_path, model_path):
    pipeline = dai.Pipeline()
    model_config = load_config(config_path)
    nnConfig = model_config.get("nn_config", {})
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    # Камера
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # YOLO сеть
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")

    # Выход для preview кадров
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb_preview")

    # Настройки YOLO
    detectionNetwork.setConfidenceThreshold(float(confidenceThreshold))
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(np.array(anchors))
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(float(iouThreshold))
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Связи
    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.out.link(nnOut.input)
    detectionNetwork.passthrough.link(xoutRgb.input)  # Важно: возвращает входной кадр

    return pipeline



pipeline = create_camera_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)

# Подключение к устройству
with dai.Device(pipeline) as device:
    # Очереди для детекций и preview
    detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)
    previewNN = device.getOutputQueue("rgb_preview", maxSize=4, blocking=False)

    # Видео
    fps = 30
    frame_width, frame_height = CAMERA_PREVIEW_DIM
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0

    while True:
        inPreview = previewNN.tryGet()

        if inPreview is not None:
            frame = inPreview.getCvFrame()
            frame_count += 1

            # Получаем детекции
            inDet = detectionNN.tryGet()
            detections = []
            if inDet is not None:
                detections = inDet.detections
                print(f"Detections: {len(detections)}")

            # Отрисовка детекций
            for detection in detections:
                if detection.confidence > 0.5:
                    # Нормализованные координаты -> пиксели
                    x1 = int(detection.xmin * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    x2 = int(detection.xmax * frame.shape[1])
                    y2 = int(detection.ymax * frame.shape[0])

                    # Правильные атрибуты DepthAI ImgDetection
                    class_id = int(detection.label)
                    confidence = detection.confidence
                    label_text = f"{LABELS[class_id]}: {confidence:.2f}"

                    # Рисуем
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Сохранение кадра в видео
            out.write(frame)

            # FPS
            if frame_count % 30 == 0:
                end_time = time.time()
                fps_actual = 30 / (end_time - start_time)
                print(f"Actual FPS: {fps_actual:.2f}")
                start_time = end_time


            cv2.imshow("OAK-D YOLO Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break


    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_VIDEO}")
