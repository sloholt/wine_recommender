# pip install google-cloud-vision
import sys
from google.cloud import vision


def read_label(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as f:
        image = vision.Image(content=f.read())

    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    # The first annotation holds all the text found on the label
    return response.text_annotations[0].description if response.text_annotations else ""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python read_label.py <image_path>")
    print(read_label(sys.argv[1]))
