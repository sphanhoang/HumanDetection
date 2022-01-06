import cv2 as cv
import numpy as np
COLOR = (0,255,0)

def detect(frame, net, index):
    results = []
    height, width = frame.shape[:2]

    # Break the frame into 4-dimensional blobs for yolo input,
    # resize without cropping (to 320x320), also swap BGR to RGB
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv.imshow(str(n), img_blob)

    # pass the blobs through yolo model
    net.setInput(blob)

    # get the output layers' names and pass layers' names into layerOutput array
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutput = net.forward(output_layers_names)

    boxes = []
    confidences = []
    # class_ids = []

    # loop over each of layer outputs
    for output in layerOutput:
        # loop over each of bounding box arrays (detection box):
        for detection in output:
            # in the bounding box array, from the 5th elements onward are the
            # confident scores of each type of object
            # So we extract those confidence scores from the bounding box array
            scores = detection[5:]
            # We take the highest score's index (the nth element)
            class_id = np.argmax(scores)
            # as well as the highest score itself
            confidence = scores[class_id]
            # if the score's index (nth element) is also the index of "people" type
            # in classes[] array which is the first element (0), then we take
            # that bounding box. Also we only take if confident score greater than 0.5
            if (class_id == index) & (confidence > 0.7):
                # The first 4 elements in bounding box array are:
                # x center, y center, box's width, box's height respectively
                # We have to scale bounding box's coordinates back relative
                # to the size of original frame.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # x, y being the coordinates of top left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # save the results into arrays:
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                # class_ids.append(class_id)

    # non-maximum suppression (NMS) to get rid of weak overlapping boxes
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

    # Ensure that at least 1 detection exists
    # then save detection boxes' parameters
    centroids, groundpoints, rectangles = [], [], []
    if len(indexes) > 0:
        for i in indexes.flatten():
            # extract the bounding box's coordinates:
            x, y, w, h = boxes[i]
            centroid = int(x+w/2), int(y+h/2)
            groundpoint = int(x+w/2), int(y+h)
            rectangle = x, y, int(x+w), int(y+h)
            # save results:
            centroids.append(centroid)
            groundpoints.append(groundpoint)
            rectangles.append(rectangle)
    return centroids, groundpoints, rectangles


##################################################################################################
if __name__ == "__main__":
    # Load YOLOv4 model and enable GPU
    yolo = cv.dnn.readNetFromDarknet(".\yolo\yolov4-tiny.cfg", ".\yolo\yolov4-tiny.weights")
    # yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Load objects names. Later when we have our own trained model we do not need this
    classes = []
    with open("yolo/coco.names", 'r') as f:
        classes = f.read().splitlines()
    # print(classes)

    cap = cv.VideoCapture(0) # change to 0 for live webcam
    while True:
        isTrue, frame = cap.read()
        centroids, groundpoints, rectangles = detect(frame, yolo, index=classes.index("person"))
        for i in range(0, len(rectangles)):
            topleft_X, topleft_Y, botright_X, botright_Y = rectangles[i]
            cv.rectangle(frame, (topleft_X, topleft_Y), (botright_X, botright_Y), COLOR, 2)
            cv.circle(frame, centroids[i], 5, COLOR, -1)
        cv.imshow('Video', frame)
        if cv.waitKey(16) & 0xFF==ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
