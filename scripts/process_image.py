from ultralytics import YOLO
import cv2
import numpy as np
import colorsys
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import base64
# import json
# import uuid
import os

def calculate_rgb_color(index):
    degree = 180.0 * (index % 2) + 90.0 * ((index // 2) % 2) + 45.0 * ((index // 4) % 2) + 22.5 * ((index // 8) % 2)
    hsv_color = (degree / 360.0, 0.5, 0.5)
    rgb_color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(*hsv_color))
    return rgb_color


def convert_rgb_to_bgr(rgb_color):
    return rgb_color[2], rgb_color[1], rgb_color[0]


def transform_deepface_analysis(data):
    item = data[0]
    transformed = {
        "age": item["age"],
        "gender": "male" if item["dominant_gender"] == "Man" else "female",
        "race": item["dominant_race"]
    }

    return transformed


# def draw_bounding_boxes(np_ndarray, json_data):
#     # Iterate over each face data in the JSON
#     for face_data in json_data:
#         region = face_data['region']
#         x, y, w, h = region['x'], region['y'], region['w'], region['h']
#
#         # Draw rectangle around the face
#         cv2.rectangle(np_ndarray, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#     # Save the image with bounding boxes
#     cv2.imwrite(f'image_with_bounding_boxes_{uuid.uuid4()}.jpg', np_ndarray)


def deepface_analyze(np_ndarray: np.ndarray):
    try:
        analysis = DeepFace.analyze(
            img_path=np_ndarray,
            actions=['age', 'gender', 'race'],
            detector_backend='yolov8'
        )
        # draw_bounding_boxes(np_ndarray, analysis)
        return transform_deepface_analysis(analysis)
    except Exception as e:
        print(f"Failed to analyze person with error: {e}")
        return dict({
            "age": -1,
            "gender": '',
            "race": ''
        })


def get_mask_and_analysis(original_image, yolo_segmentation):
    orig_height, orig_width = original_image.shape[:2]

    # Create a white background image of the same size as the original image
    image = np.ones((orig_height, orig_width, 3), dtype=np.uint8) * 255

    # Get bounding boxes and sort them from left to right
    boxes = yolo_segmentation.boxes.xyxy
    sorted_indices = boxes[:, 0].argsort()
    deepface_analysis_results = []

    # Apply each mask
    for i, index in enumerate(sorted_indices):
        if yolo_segmentation.boxes[index].cls.item() != 0:
            continue

        mask_obj = yolo_segmentation.masks[index]

        # Convert the PyTorch tensor to a numpy array
        mask = mask_obj.data.cpu().numpy()

        # Resize mask to match the original image dimensions
        # Note: mask[0] is used to select the first channel if the mask has multiple channels
        mask_resized = cv2.resize(mask[0], (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

        # Convert the resized mask to a boolean array
        mask_bool = mask_resized.astype(bool)

        # Create a color layer based on the mask
        color_layer = np.zeros(image.shape, dtype=np.uint8)
        # color_layer[:, :] = convert_rgb_to_bgr(calculate_rgb_color(i))
        color_layer[:, :] = calculate_rgb_color(i)

        # Apply the color layer where the mask is
        image[mask_bool] = color_layer[mask_bool]

        # Create a black image of the same size as the original image
        cropped_image = np.zeros_like(image)

        # Apply the mask to the original image
        cropped_image[mask_bool] = original_image[mask_bool]

        # Find the bounding box of the mask
        y_indices, x_indices = np.where(mask_bool)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Crop the image to the bounding box of the mask
        cropped_image = cropped_image[y_min:y_max, x_min:x_max]

        # Convert the cropped image into np.ndarray
        cropped_image = np.array(cropped_image)

        # Display the cropped image
        # cv2.imshow('Cropped Image1' + str(index), image)
        # cv2.imshow('Cropped Image2' + str(index), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        deepface_analysis_results.append(deepface_analyze(cropped_image))

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), deepface_analysis_results


def load_image(image_or_path):
    image = None
    if isinstance(image_or_path, str):
        try:
            # Try to load image from path
            image = Image.open(image_or_path)
        except Exception as openException:
            try:
                if image_or_path.startswith("data:image/"):
                    image_or_path = image_or_path.split(";", maxsplit=1)[1].split(",", maxsplit=1)[1]
                image = Image.open(BytesIO(base64.b64decode(image_or_path)))
            except Exception as base64Exception:
                print(f"Failed to load image from path with error: {openException}")
                print(f"Failed to decode image from base64 string error: {base64Exception}")
                raise Exception("Image path is invalid or base64 string is invalid.")
        image = np.array(image)
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise Exception("Invalid input. Please provide a valid image path, base64 string, or numpy array.")

    return image


def get_yolo_segmentation(image):
    ultralytics_dir = os.getenv('ULTRALYTICS_HOME', '')
    model = YOLO(model=os.path.join(ultralytics_dir, "yolov8x-seg.pt"))
    results = model([image])
    # results[0].show()
    return results[0]


def process(image_or_path):
    image = load_image(image_or_path)
    yolo_segmentation = get_yolo_segmentation(image)
    return get_mask_and_analysis(image, yolo_segmentation)


# print(process('img8.png'))
