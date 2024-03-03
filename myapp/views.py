from django.shortcuts import render, redirect
from django.templatetags.static import static
from .models import Image
from .forms import ImageUploadForm, ClusterForm
import os
import base64
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from multiprocessing import Pool, cpu_count
import cv2
from sklearn.cluster import KMeans, DBSCAN
from skimage.color import rgb2hsv, hsv2rgb
from skimage.io import imread
from mtcnn.mtcnn import MTCNN
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from sklearn.svm import SVC
import face_recognition
import PIL
import matplotlib
import mediapipe as mp
from torchvision import models
from torchvision.utils import draw_segmentation_masks, draw_keypoints
import torch


matplotlib.use("Agg")

DEFAULT_IMAGE_PATH = static("images/default.png")


def rcnnImageKeyPoints(image_path):
    weights = models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.keypointrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    transforms = weights.transforms()
    person_int = read_image(image_path)
    person_float = transforms(person_int)

    outputs = model([person_float])
    kpts = outputs[0]["keypoints"]
    scores = outputs[0]["scores"]

    detect_threshold = 0.75
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]
    res = draw_keypoints(person_int, keypoints, colors="blue", radius=3)

    connect_skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]
    res = draw_keypoints(
        person_int,
        keypoints,
        connectivity=connect_skeleton,
        colors="blue",
        radius=4,
        width=3,
    )

    image = np.array(to_pil_image(res))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def maskrcnnImageSegmentation(image_path):
    weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    transforms = weights.transforms()

    image = read_image(image_path)
    imageT = transforms(image)

    output = model([imageT])
    person_output = output[0]

    proba_threshold = 0.5
    person_bool_masks = person_output["masks"] > proba_threshold
    person_bool_masks = person_bool_masks.squeeze(1)

    score_threshold = 0.75
    boolean_masks = (
        output[0]["masks"][output[0]["scores"] > score_threshold] > proba_threshold
    )
    person_with_masks = draw_segmentation_masks(image, boolean_masks.squeeze(1))

    image = np.array(to_pil_image(person_with_masks))

    return image


def lrasppImageSegmentation(image_path):
    weights = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    img = read_image(image_path)
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["person"]]

    image = np.array(to_pil_image(mask))

    return image


def deepLabImageSegmentation(image_path):
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    img = read_image(image_path)
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["person"]]

    image = np.array(to_pil_image(mask))

    return image


def mediapipeFaceDetection(image_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

    return image


def mediapipeFaceMesh(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

    return image


def deeplearningDlibFaceDetection(image_path):
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            draw.line(face_landmarks[facial_feature], width=5)

    output = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return output


def fcnImageSegmentation(image_path):
    weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT
    model = models.segmentation.fcn_resnet101(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    img = read_image(image_path)
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["person"]]

    image = np.array(to_pil_image(mask))

    return image


def haarCascadeFaceDetection(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return img


def mtcnnFaceDetection(image_path):
    source = cv2.imread(image_path, cv2.IMREAD_COLOR)

    model = MTCNN()
    output = model.detect_faces(source)

    face_count = len(output)

    for i in range(0, face_count):
        conf = output[i]["confidence"]

        if conf > 0.6:
            box = output[i]["box"]
            keypoints = output[i]["keypoints"]
            left_eye = list(keypoints["left_eye"])
            right_eye = list(keypoints["right_eye"])
            nose = list(keypoints["nose"])
            mouth_right = list(keypoints["mouth_right"])
            mouth_left = list(keypoints["mouth_left"])

            cv2.rectangle(
                source,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (0, 0, 255),
                4,
            )
            cv2.rectangle(
                source,
                (left_eye[0] - 10, left_eye[1] - 10),
                (left_eye[0] + 10, left_eye[1] + 10),
                (0, 0, 255),
                4,
            )
            cv2.rectangle(
                source,
                (right_eye[0] - 10, right_eye[1] - 10),
                (right_eye[0] + 10, right_eye[1] + 10),
                (0, 0, 255),
                4,
            )
            cv2.rectangle(
                source,
                (nose[0] - 25, nose[1] - 25),
                (nose[0] + 25, nose[1] + 25),
                (0, 0, 255),
                4,
            )
            cv2.rectangle(
                source,
                (mouth_right[0], mouth_right[1] - 5),
                (mouth_left[0], mouth_left[1]),
                (0, 0, 255),
                4,
            )

    return source


def hsvThresholdSegmentation(image_path):
    image = imread(image_path)
    image_hsv = rgb2hsv(image)
    image_h = image_hsv[:, :, 0]
    n, bins, patches = plt.hist(image_h.ravel(), bins=256, range=[0, 1])
    peaks, _ = find_peaks(n, height=0)
    if peaks.size > 0:
        peak_hue = bins[peaks[0]]

        range_width = 0.075

        lower_bound = max(0, peak_hue - range_width)
        upper_bound = min(1, peak_hue + range_width)

        lower_mask = image_hsv[:, :, 0] > lower_bound
        upper_mask = image_hsv[:, :, 0] < upper_bound
        mask = lower_mask * upper_mask

        image_red = image[:, :, 0] * mask
        image_green = image[:, :, 1] * mask
        image_blue = image[:, :, 2] * mask

        segmented_image = np.dstack((image_red, image_green, image_blue))

        return segmented_image


def svmSegmentation(image_path):
    source = cv2.imread(image_path)

    original_height, original_width = source.shape[:2]
    if original_height > original_width:
        ratio = 256 / original_height
        new_width = int(original_width * ratio)
        source = cv2.resize(source, (new_width, 256), interpolation=cv2.INTER_AREA)
    else:
        ratio = 256 / original_width
        new_height = int(original_height * ratio)
        source = cv2.resize(source, (256, new_height), interpolation=cv2.INTER_AREA)

    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    M, N, dim = source.shape

    im_hsv = rgb2hsv(source)

    X = np.reshape(im_hsv, (-1, dim))
    Y = np.zeros(X.shape[0])
    Y = (X[:, 0] < 0.2).astype(int)

    svm = SVC(probability=True)
    svm.fit(X, Y)

    pred_probs = svm.predict_proba(X)
    predX = pred_probs[:, 1] > 0.5
    predX = np.reshape(predX, im_hsv.shape[:2])

    im_pred = np.zeros((M, N, dim))
    im_pred[..., 0] = predX
    im_pred[..., 1] = predX
    im_pred[..., 2] = predX
    im_pred_rgb = hsv2rgb(im_pred)

    im_pred_rgb = cv2.cvtColor(
        cv2.cvtColor((im_pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2RGB,
    )
    return im_pred_rgb


def cannyEdgeDetection(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    gray_blurred = ndimage.gaussian_filter(gray, sigma=1.4)

    def Normalize(img):
        img = img / np.max(img)
        return img

    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    gx = ndimage.convolve(gray_blurred, Gx)
    gx = Normalize(gx)

    Gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    gy = ndimage.convolve(gray_blurred, Gy)
    gy = Normalize(gy)

    Mag = np.hypot(gx, gy)
    Mag = Normalize(Mag)

    Gradient = np.degrees(np.arctan2(gy, gx))

    NMS = np.zeros(Mag.shape)

    for i in range(1, int(Mag.shape[0]) - 1):
        for j in range(1, int(Mag.shape[1]) - 1):
            if (Gradient[i, j] >= 0 and Gradient[i, j] <= 45) or (
                Gradient[i, j] < -135 and Gradient[i, j] >= -180
            ):
                yBot = np.array([Mag[i, j + 1], Mag[i + 1, j + 1]])
                yTop = np.array([Mag[i, j - 1], Mag[i - 1, j - 1]])
                x_est = np.absolute(gy[i, j] / Mag[i, j])
                if Mag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Mag[
                    i, j
                ] >= ((yTop[1] - yTop[0]) * x_est + yTop[0]):
                    NMS[i, j] = Mag[i, j]
                else:
                    NMS[i, j] = 0
            if (Gradient[i, j] > 45 and Gradient[i, j] <= 90) or (
                Gradient[i, j] < -90 and Gradient[i, j] >= -135
            ):
                yBot = np.array([Mag[i + 1, j], Mag[i + 1, j + 1]])
                yTop = np.array([Mag[i - 1, j], Mag[i - 1, j - 1]])
                x_est = np.absolute(gx[i, j] / Mag[i, j])
                if Mag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Mag[
                    i, j
                ] >= ((yTop[1] - yTop[0]) * x_est + yTop[0]):
                    NMS[i, j] = Mag[i, j]
                else:
                    NMS[i, j] = 0
            if (Gradient[i, j] > 90 and Gradient[i, j] <= 135) or (
                Gradient[i, j] < -45 and Gradient[i, j] >= -90
            ):
                yBot = np.array([Mag[i + 1, j], Mag[i + 1, j - 1]])
                yTop = np.array([Mag[i - 1, j], Mag[i - 1, j + 1]])
                x_est = np.absolute(gx[i, j] / Mag[i, j])
                if Mag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Mag[
                    i, j
                ] >= ((yTop[1] - yTop[0]) * x_est + yTop[0]):
                    NMS[i, j] = Mag[i, j]
                else:
                    NMS[i, j] = 0
            if (Gradient[i, j] > 135 and Gradient[i, j] <= 180) or (
                Gradient[i, j] < 0 and Gradient[i, j] >= -45
            ):
                yBot = np.array([Mag[i, j - 1], Mag[i + 1, j - 1]])
                yTop = np.array([Mag[i, j + 1], Mag[i - 1, j + 1]])
                x_est = np.absolute(gy[i, j] / Mag[i, j])
                if Mag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Mag[
                    i, j
                ] >= ((yTop[1] - yTop[0]) * x_est + yTop[0]):
                    NMS[i, j] = Mag[i, j]
                else:
                    NMS[i, j] = 0

    NMS = Normalize(NMS)

    highThresholdRatio = 0.2
    lowThresholdRatio = 0.15
    GSup = np.copy(NMS)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    x = 0.1
    oldx = 0

    while oldx != x:
        oldx = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if GSup[i, j] > highThreshold:
                    GSup[i, j] = 1
                elif GSup[i, j] < lowThreshold:
                    GSup[i, j] = 0
                else:
                    if (
                        (GSup[i - 1, j - 1] > highThreshold)
                        or (GSup[i - 1, j] > highThreshold)
                        or (GSup[i - 1, j + 1] > highThreshold)
                        or (GSup[i, j - 1] > highThreshold)
                        or (GSup[i, j + 1] > highThreshold)
                        or (GSup[i + 1, j - 1] > highThreshold)
                        or (GSup[i + 1, j] > highThreshold)
                        or (GSup[i + 1, j + 1] > highThreshold)
                    ):
                        GSup[i, j] = 1
        x = np.sum(GSup == 1)

    GSup = (GSup == 1) * GSup
    cannyImage = cv2.cvtColor(
        cv2.cvtColor((GSup * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2RGB,
    )

    return cannyImage


def cannyEdgeDetection1(gray):
    blur_gray = gaussianBlurImage(gray)
    mean_intensity = np.mean(blur_gray)
    std_intensity = np.std(blur_gray)
    k = 0.5

    low_threshold = int(max(0, (1.0 - k) * mean_intensity - k * std_intensity))
    high_threshold = int(min(255, (1.0 + k) * mean_intensity + k * std_intensity))

    canny = cv2.cvtColor(
        cv2.Canny(blur_gray, low_threshold, high_threshold), cv2.COLOR_GRAY2RGB
    )

    return canny


def gaussianBlurImage(image):
    GaussianBlur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    GaussianBlur = cv2.filter2D(src=image, ddepth=-1, kernel=GaussianBlur_kernel)

    return GaussianBlur


def scharrEdgeDetection(image):
    normalized_img = image / 255
    scharrx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32)

    Ix = ndimage.filters.convolve(normalized_img, scharrx)
    scharry = np.rot90(scharrx)
    Iy = ndimage.filters.convolve(normalized_img, scharry)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G


def sobelEdgeDetection(image):
    normalized_img = image / 255
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

    Ix = ndimage.filters.convolve(normalized_img, sobelx)
    sobely = np.rot90(sobelx)
    Iy = ndimage.filters.convolve(normalized_img, sobely)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def prewittEdgeDetection(gray):
    normalized_img = gray / 255
    prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

    Ix = ndimage.filters.convolve(normalized_img, prewittx)
    Iy = ndimage.filters.convolve(normalized_img, prewitty)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    return G


def robertsEdgeDetection(gray):
    normalized_img = gray / 255
    robertsx = np.array([[1, 0], [0, -1]], np.float32)
    robertsy = np.array([[0, 1], [-1, 0]], np.float32)

    Ix = ndimage.filters.convolve(normalized_img, robertsx)
    Iy = ndimage.filters.convolve(normalized_img, robertsy)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    return G


def encodeImage(image):
    _, buffer = cv2.imencode(".jpg", image)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image


def my_kmeans(data, k, criteria, max_iterations=100, flags=None):
    centroids = data[np.random.choice(len(data), k, replace=False)]

    compactness_history = []

    for _ in range(max_iterations):
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)

        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)

        compactness = np.sum((data - centroids[labels]) ** 2)
        compactness_history.append(compactness)

    return compactness_history, labels, centroids


def kmeansSegmentation(image_path, num_of_clusters):
    source = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = source.shape
    if height > 1000 or width > 1000:
        source = cv2.resize(source, (0, 0), fx=0.25, fy=0.25)
    source_2d = source.reshape((-1, 3))
    source_2d = np.float32(source_2d)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.85)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = my_kmeans(
        source_2d, num_of_clusters, criteria, 10, flags
    )

    centers = np.uint8(centers)
    output = centers[labels.flatten()]
    output = output.reshape((source.shape))

    return output


def dbscanSegmentation(image_path):
    eps = 10
    min_samples = 100

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, dim = image.shape

    original_height, original_width = image.shape[:2]
    if original_height > original_width:
        ratio = 256 / original_height
        new_width = int(original_width * ratio)
        image = cv2.resize(image, (new_width, 256), interpolation=cv2.INTER_AREA)
    else:
        ratio = 256 / original_width
        new_height = int(original_height * ratio)
        image = cv2.resize(image, (256, new_height), interpolation=cv2.INTER_AREA)
    img_flat = image.reshape((-1, 3))

    height, width = image.shape[:2]
    y_positions, x_positions = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    pixels_positions = np.column_stack((y_positions.flatten(), x_positions.flatten()))

    features = np.hstack((img_flat, pixels_positions))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    clustered_img = clusters.reshape(image.shape[:2])

    im_pred = np.zeros((height, width, dim))
    im_pred[..., 0] = clustered_img
    im_pred[..., 1] = clustered_img
    im_pred[..., 2] = clustered_img
    im_pred_rgb = hsv2rgb(im_pred)

    im_pred_rgb = cv2.cvtColor(
        cv2.cvtColor((im_pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2RGB,
    )

    return im_pred_rgb


def logEdgeDetection(gray):
    img_blur = gaussianBlurImage(gray)

    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    laplacian = cv2.filter2D(img_blur, cv2.CV_64F, laplacian_kernel)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    log_edges = np.abs(laplacian_abs).astype(np.uint8)

    return log_edges


def laplacianEdgeDetection(gray):
    img_blur = gaussianBlurImage(gray)

    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    laplacian = cv2.filter2D(img_blur, cv2.CV_64F, laplacian_kernel)
    laplacian_abs = np.uint8(np.absolute(laplacian))

    return laplacian_abs


def sharpenImage(gray):
    edge_kernal = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    identity_kernal = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    sharpen_kernal = edge_kernal + identity_kernal
    sharp_img = cv2.filter2D(gray, -1, sharpen_kernal)

    return sharp_img


def grayscaleImage(image_path):
    img = cv2.imread(image_path, 0)
    height, width = img.shape
    if height > 1000 or width > 1000:
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    return img


def thresholdImage(gray):
    threshold_value = 128
    thresholded_image = np.where(gray > threshold_value, 255, 0).astype(np.uint8)

    return thresholded_image


def adaptiveThresholdImage(image_path):
    image = plt.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_img = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5
    )

    return thresholded_img


def checkClusters(image_path):
    source = cv2.imread(image_path)
    height, width, _ = source.shape
    if height > 1000 or width > 1000:
        source = cv2.resize(source, (0, 0), fx=0.25, fy=0.25)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    source_2d = source.reshape((-1, 3))
    source_2d = np.float32(source_2d)

    inertia = []

    for i in range(1, 10):
        km = KMeans(n_clusters=i, init="k-means++")
        km.fit(source_2d)
        inertia.append(km.inertia_)

    df = pd.DataFrame({"Inertia": inertia, "Clusters": range(1, 10)})

    fig, ax = plt.subplots()

    ax.plot(df["Clusters"], df["Inertia"])
    ax.scatter(df["Clusters"], df["Inertia"])

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    return image_base64


def worker(task_info):
    func, args = task_info
    return func(*args)


def myapp(request):
    uploaded_images = Image.objects.all()

    folder_path = "media/images"
    existing_images = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    images = list(uploaded_images) + [{"filename": f} for f in existing_images]

    selected_image = None
    num_of_clusters = 3
    image_id = request.GET.get("select_image")
    if image_id:
        try:
            image_id = int(image_id)
            if image_id < len(uploaded_images):
                Image.objects.update(selected=False)
                selected_image = uploaded_images[image_id]
                selected_image.selected = True
                selected_image.save()
            else:
                fs_image_index = image_id - len(uploaded_images)
                if fs_image_index < len(existing_images):
                    selected_image = existing_images[fs_image_index]
        except ValueError:
            pass

    if request.method == "POST":
        cluster_form = ClusterForm(request.POST)
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            Image.objects.update(selected=False)
            image_instance = Image(
                title=form.cleaned_data["image"].name, image=form.cleaned_data["image"]
            )
            image_instance.selected = True
            image_instance.save()
            return redirect("myapp")
        if cluster_form.is_valid():
            num_of_clusters = int(cluster_form.cleaned_data["cluster_input"])
        else:
            num_of_clusters = 3
    else:
        form = ImageUploadForm()
        cluster_form = ClusterForm()

    if selected_image:
        image_path = selected_image.image.path

        grayImage = grayscaleImage(image_path)
        sharpImage = sharpenImage(grayImage)
        sobelImage, theta = sobelEdgeDetection(grayImage)

        tasks = [
            (gaussianBlurImage, (grayImage,)),
            (laplacianEdgeDetection, (sharpImage,)),
            (scharrEdgeDetection, (grayImage,)),
            (cannyEdgeDetection, (image_path,)),
            (cannyEdgeDetection1, (grayImage,)),
            (robertsEdgeDetection, (grayImage,)),
            (prewittEdgeDetection, (grayImage,)),
            (logEdgeDetection, (grayImage,)),
            (thresholdImage, (grayImage,)),
            (adaptiveThresholdImage, (image_path,)),
            (checkClusters, (image_path,)),
            (
                kmeansSegmentation,
                (
                    image_path,
                    int(num_of_clusters),
                ),
            ),
            (dbscanSegmentation, (image_path,)),
            (svmSegmentation, (image_path,)),
            (mtcnnFaceDetection, (image_path,)),
            (haarCascadeFaceDetection, (image_path,)),
            (fcnImageSegmentation, (image_path,)),
            (deeplearningDlibFaceDetection, (image_path,)),
            (mediapipeFaceDetection, (image_path,)),
            (mediapipeFaceMesh, (image_path,)),
            (deepLabImageSegmentation, (image_path,)),
            (lrasppImageSegmentation, (image_path,)),
            (maskrcnnImageSegmentation, (image_path,)),
            (rcnnImageKeyPoints, (image_path,)),
            (hsvThresholdSegmentation, (image_path,)),
        ]
        num_processes = min(len(tasks), cpu_count())
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker, tasks)

        (
            gaussianImage,
            laplacianImage,
            scharrImage,
            cannyImage,
            cannyImage1,
            robertsImage,
            prewittImage,
            logImage,
            threshImage,
            adaptiveThreshImage,
            clusterGraph,
            kmeansImage,
            dbscanImage,
            svmImage,
            mtcnnImage,
            haarCascadeImage,
            fcnSegmentationImage,
            dlibImage,
            blazeFaceImage,
            faceMeshImage,
            deepLabImage,
            lrasppImage,
            maskrcnnImage,
            keypointImage,
            hsvSegmentationImage,
        ) = results
    else:
        defaultImage = cv2.imread("myapp" + DEFAULT_IMAGE_PATH)
        selected_image = encodeImage(defaultImage)
        grayImage = defaultImage
        gaussianImage = defaultImage
        sharpImage = defaultImage
        laplacianImage = defaultImage
        kmeansImage = defaultImage
        sobelImage = defaultImage
        scharrImage = defaultImage
        cannyImage = defaultImage
        cannyImage1 = defaultImage
        robertsImage = defaultImage
        prewittImage = defaultImage
        logImage = defaultImage
        threshImage = defaultImage
        adaptiveThreshImage = defaultImage
        clusterGraph = encodeImage(defaultImage)
        dbscanImage = defaultImage
        svmImage = defaultImage
        mtcnnImage = defaultImage
        haarCascadeImage = defaultImage
        fcnSegmentationImage = defaultImage
        dlibImage = defaultImage
        blazeFaceImage = defaultImage
        faceMeshImage = defaultImage
        deepLabImage = defaultImage
        lrasppImage = defaultImage
        maskrcnnImage = defaultImage
        keypointImage = defaultImage
        hsvSegmentationImage = defaultImage

    return render(
        request,
        "image_app/myapp.html",
        {
            "images": list(enumerate(images)),
            "form": form,
            "selected_image": selected_image,
            "gray_image": encodeImage(grayImage),
            "gaussian_image": encodeImage(gaussianImage),
            "sharp_image": encodeImage(sharpImage),
            "laplacian_edge_detection": encodeImage(laplacianImage),
            "sobel_image": encodeImage(sobelImage),
            "scharr_image": encodeImage(scharrImage),
            "canny_image": encodeImage(cannyImage1),
            "canny_image_1": encodeImage(cannyImage),
            "roberts_image": encodeImage(robertsImage),
            "prewitt_image": encodeImage(prewittImage),
            "log_image": encodeImage(logImage),
            "thresh_image": encodeImage(threshImage),
            "adaptive_thresh_image": encodeImage(adaptiveThreshImage),
            "cluster_graph": clusterGraph,
            "kmeans_segmentation": encodeImage(kmeansImage),
            "dbscan_image": encodeImage(dbscanImage),
            "svm_image": encodeImage(svmImage),
            "mtcnn_image": encodeImage(mtcnnImage),
            "haar_cascade_image": encodeImage(haarCascadeImage),
            "fcn_segmentation": encodeImage(fcnSegmentationImage),
            "hsv_segmentation": encodeImage(hsvSegmentationImage),
            "dlib_image": encodeImage(dlibImage),
            "blaze_face_image": encodeImage(blazeFaceImage),
            "face_mesh_image": encodeImage(faceMeshImage),
            "deeplab_image": encodeImage(deepLabImage),
            "lraspp_image": encodeImage(lrasppImage),
            "maskrcnn_image": encodeImage(maskrcnnImage),
            "keypoint_image": encodeImage(keypointImage),
        },
    )
