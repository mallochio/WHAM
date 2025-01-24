import os
import base64
import argparse

from openai import OpenAI
import cv2
from tqdm import tqdm

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-1.5-flash-8b"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.2-90b-vision-preview"

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/"
MISTRAL_MODEL = "pixtral-large-latest"

QUERY = """
Analyze this overhead fisheye camera image with person segmentation masks:
- The image is from a ceiling-mounted omnidirectional camera
- Person segmentation masks are overlaid in semi-transparent color
- Task: Check if ALL persons in the image have corresponding masks
- Response format: 
    * Return 'YES' if every person has a mask
    * Return 'NO' if any person is missing a mask
- Strictly respond with only 'YES' or 'NO', no other text
"""

MASK_DIR = "/home/NAS-mountpoint/kinect-omni-ego/2022-09-29/at-a02/kitchen/a03/omni_masks/out_mask"
IMG_DIR = "/home/NAS-mountpoint/kinect-omni-ego/2022-09-29/at-a02/kitchen/a03/omni"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def overlay_mask(image_path, mask_path, output_dir=None):
    # Overlay the mask on the image and optionally saves
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    if mask is not None and img is not None:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1, mask, alpha, 0)
        if output_dir is not None:
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, overlay)
    return overlay


def make_overlays(image_dir=IMG_DIR, mask_dir=MASK_DIR):
    """
    Takes in directories containing masks and images and calls function that overlays the masks on the images and saves
    """
    # Get the list of images
    images = os.listdir(image_dir)
    output_dir = os.path.join(os.path.dirname(mask_dir), "out_composite")
    for image in tqdm(images):
        image_path = os.path.join(image_dir, image)
        mask_path = os.path.join(mask_dir, image)
        if not os.path.exists(mask_path):
            continue
        overlay_mask(image_path, mask_path, output_dir=output_dir)


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
    response = response.choices[0].message.content
    return response


def loop_over_dir(args, image_dir=IMG_DIR, mask_dir=MASK_DIR):
    api_key = args.api_key
    False_negatives = []
    for i in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, i)
        mask_path = os.path.join(mask_dir, i)
        if not os.path.exists(mask_path):
            response = call_api(api_key, image_path)
            if response == "NO":
                False_negatives.append(image_path)
    print(f"False negatives: {False_negatives}")

def main(args):
    loop_over_dir(args)
    call_api(args.api_key, args.image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API call example")
    parser.add_argument("--api_key", required=True, help="Your API key")
    parser.add_argument("--image_path", help="Path to the image file", default="/home/NAS-mountpoint/kinect-omni-ego/2022-09-29/at-a02/kitchen/a03/omni_masks/out_composite/1664459293715.jpg")
    parser.add_argument("--query", required=False, help="Query to ask the model")
    args = parser.parse_args()
    main(args)
    # make_overlays()
