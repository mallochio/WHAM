import os
import base64
import argparse

from matplotlib.pyplot import legend
from matplotlib.quiver import QuiverKey
from openai import OpenAI
import cv2
import numpy as np
from tqdm import tqdm

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-1.5-flash-8b"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.2-90b-vision-preview"

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/"
MISTRAL_MODEL = "pixtral-12b-2409"

QUERY = """
This is a photo captured from a ceiling mounted omnidirectional camera with a fisheye lens. \
I have also overlaid a segmentation mask of a person on top of the image, \
so any persons in the image are supposed to have masks overlaid on them. \
If a person is missing a mask, reply with a single word - NO. \
If no persons are missed - reply YES \
Do not output any other text, only YES or NO.\
"""

MASK_DIR = "/home/NAS-mountpoint/kinect-omni-ego/2022-09-29/at-a02/kitchen/a03/omni_masks/out_mask"
IMG_DIR = "/home/NAS-mountpoint/kinect-omni-ego/2022-09-29/at-a02/kitchen/a03/omni"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def make_overlay(image_dir=IMG_DIR, mask_dir=MASK_DIR):
    # Get the list of images
    images = os.listdir(image_dir)
    output_dir = os.path.join(os.path.dirname(mask_dir), "out_composite")
    for image in tqdm(images):
        image_path = os.path.join(image_dir, image)
        mask_path = os.path.join(mask_dir, image)
        if not os.path.exists(mask_path):
            print(image_path)
            continue

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        if mask is not None and img is not None:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            alpha = 0.5
            overlay = cv2.addWeighted(img, 1, mask, alpha, 0)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, overlay)


def call_api(api_key, image_path, query=QUERY):
    client = OpenAI(
        api_key=api_key,
        base_url=MISTRAL_ENDPOINT
    )
    # Getting the base64 string
    base64_image = encode_image(image_path=image_path)
    response = client.chat.completions.create(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        temperature=0.5,
        top_p=0.5,
        max_tokens=256,
        # stream=False,
        # stop=None
    ) 
    print(response.choices[0].message.content)

def main(args):
    call_api(args.api_key, args.image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API call example")
    parser.add_argument("--api_key", required=True, help="Your API key")
    parser.add_argument("--image_path", required=True, help="Path to the image file")
    parser.add_argument("--query", required=False, help="Query to ask the model")
    args = parser.parse_args()
    # main(args)
    make_overlay()
