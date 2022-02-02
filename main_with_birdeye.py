import cv2 as cv
import numpy as np
from scipy.spatial import distance
from Detector import detect

CIRCLE = 60
DOT = 3
MIN_DISTANCE = 200
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
corner_points = [(243, 357), (785, 203), (707, 713), (1152, 359)]
width_og = 1000
height_og = 750
static_img = "./frame.jpg"


def perspective_transform(corner_points, width, height, image):
    #  Compute the transformation matrix
    # @ corner_points : 4 corner points selected from the image
    # @ height, width : size of the image
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([ [0, 0], [width, 0], [0, height], [width, height] ])
    # Compute and return the transformation matrix
    matrix = cv.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed

def point_transformation(matrix, list_downoids):
    # Apply the perspective transformation to every ground point which have been detected on the main frame.
    # matrix : the 3x3 matrix
    # list_downoids : list that contains the points to transform
    # return : list containing all the new points
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return transformed_points_list

def draw_rectangle(corner_points, frame):
    # Draw rectangle box over the delimitation area
    cv.line(frame, (corner_points[0]), (corner_points[1]), COLOR_BLUE, thickness=1)
    cv.line(frame, (corner_points[1]), (corner_points[3]), COLOR_BLUE, thickness=1)
    cv.line(frame, (corner_points[0]), (corner_points[2]), COLOR_BLUE, thickness=1)
    cv.line(frame, (corner_points[3]), (corner_points[2]), COLOR_BLUE, thickness=1)

def measure(birdeye_points, MIN_DISTANCE):
    # measure distance between centroids on birdeye view
    violate = set()
    # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    if len(birdeye_points) >= 2:
        # extract all centroids from the birdeye_points[] and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([k for k in birdeye_points])
        D = distance.cdist(centroids, centroids, metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < MIN_DISTANCE:
                    # update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)
    return violate



#           MAIN            #


####################################
# Load YOLOv4 model and enable GPU #
####################################
print("[Detector] Loading YOLO...")
yolo = cv.dnn.readNetFromDarknet(".\yolo\yolov4-tiny.cfg", ".\yolo\yolov4-tiny.weights")
# yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


################################################################
# Load objects names                                           #
# Later when we have our own trained model we do not need this #
################################################################
classes = []
with open("yolo/coco.names", 'r') as f:
    classes = f.read().splitlines()

##########################################################
# Compute  transformation matrix from the original frame #
##########################################################
matrix, imgOutput = perspective_transform(corner_points, width_og, height_og, cv.imread(static_img))
height, width, _ = imgOutput.shape
blank_image = np.zeros((height, width, 3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1]
# dim = (width, height)



#################
# Capture video #
#################
print("[Detector] Capturing video stream...")
cap = cv.VideoCapture('./test.mp4')  # change to 0 for live webcam
print("[MAIN] Program is running...")
while True:
    blank = np.zeros((height,width,3), dtype='uint8')
    isTrue, frame = cap.read()
    resized = cv.resize(frame, (int(frame.shape[1] * 0.7), int(frame.shape[0] * 0.7)),
                        interpolation=cv.INTER_CUBIC)

    # get information from the dectector:
    centroids, groundpoints, rectangles = detect(resized, yolo, index=classes.index("person"))

    # transform centroids and ground points to bird eye view:
    birdeye_points = point_transformation(matrix, groundpoints)

    # measure distance between centroids on birdeye view
    violate = measure(birdeye_points, MIN_DISTANCE)

    # show every point on bird eye and main frame view:
    for i in range (0, len(birdeye_points)):
        COLOR = COLOR_GREEN
        x, y = birdeye_points[i]
        if i in violate:
            COLOR = COLOR_RED
            # print("-10 SOCIAL CREDIT")
        cv.circle(blank, (int(x), int(y)), CIRCLE, COLOR, 2)
        cv.circle(blank, (int(x), int(y)), DOT, COLOR, -1)
        topleft_X, topleft_Y, botright_X, botright_Y = rectangles[i]
        cv.rectangle(resized, (topleft_X, topleft_Y), (botright_X, botright_Y), COLOR, 2)
        cv.circle(resized, groundpoints[i], 5, COLOR, -1)
        text = "People at risk: {}".format(len(violate))
        cv.putText(resized, text, (10, resized.shape[0] - 25),
                    cv.FONT_HERSHEY_SIMPLEX, 0.85, COLOR, 3)


    # draw bird eye region on original frame:
    draw_rectangle(corner_points, resized)
    # show video:
    cv.imshow("Bird eye", blank)
    cv.imshow('Video', resized)
    if cv.waitKey(16) & 0xFF == ord('q'):
        print("[MAIN] Exiting program...")
        break
cap.release()
cv.destroyAllWindows()
