import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import models

# specify image dimension
IMAGE_SIZE = 512
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop(img, dim):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

    return crop_img


def preprocess_image(image):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (768, 768))
    image = center_crop(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    image -= MEAN
    image /= STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)

    # return the preprocessed image
    return image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path to the input image")
ap.add_argument("--arch", type=str, default="resnet50")
ap.add_argument("--path", type=str, default="")

args = ap.parse_args()

model = models.__dict__[args.arch]()
if os.path.isfile(args.path):
    print("=> loading checkpoint '{}'".format(args.path))
    if torch.cuda.is_available():
        checkpoint = torch.load(args.path, map_location='cuda')
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("module."):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# load the image from disk, clone it (so we can draw on it later),
# and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args.image)
orig = image.copy()
image = preprocess_image(image)
# convert the preprocessed image to a torch tensor and flash it to
# the current device
image = torch.from_numpy(image)
image = image.to(DEVICE)

labels = {
    0: "Bed",
    1: "Chair",
    2: "Sofa",
}

# classify the image and extract the predictions
print("[INFO] classifying image with '{}'...".format(args.arch))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)

# loop over the predictions and display the rank-5 predictions and
# corresponding probabilities to our terminal
for (i, idx) in enumerate(sortedProba[0, :3]):
    print("{}. {}: {:.2f}%".format(
        i, labels[idx.item()].strip(), probabilities[0, idx.item()] * 100
    ))


# draw the top prediction on the image and display the image to
# our screen
(label, prob) = (
    labels[probabilities.argmax().item()],
    probabilities.max().item()
)
cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("./demo.png", orig)
