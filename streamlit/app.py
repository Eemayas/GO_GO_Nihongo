import streamlit as st
import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
from transformers import MarianMTModel, MarianTokenizer
import easyocr
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import math
import pandas as pd

# Initialize OCR and translation models
reader = easyocr.Reader(["ja"], gpu=True)
ocr = MangaOcr()
model_name = "Helsinki-NLP/opus-mt-ja-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def display_image(image, caption="Image"):
    st.image(image, caption=caption, use_container_width=True)


def is_inside(small_box, large_box):
    """Check if the small bounding box is inside the large bounding box."""
    return (
        large_box[0][0] <= small_box[0][0]
        and large_box[0][1] <= small_box[0][1]
        and large_box[2][0] >= small_box[2][0]
        and large_box[2][1] >= small_box[2][1]
    )


def filter_bboxes(detected_bbox):
    """Filter out bounding boxes that are completely inside other bounding boxes."""
    filtered_bbox = []
    for i, bbox in enumerate(detected_bbox):
        inside_another = any(
            is_inside(bbox, other_bbox)
            for j, other_bbox in enumerate(detected_bbox)
            if i != j
        )
        if not inside_another:
            filtered_bbox.append(bbox)
    return filtered_bbox


def easyocr_detection(image):
    # Perform OCR on the image
    results = reader.readtext(np.array(image))

    detected_bbox = []

    # Collect results in a DataFrame-friendly format
    for bbox, text, confidence in results:
        detected_bbox.append(bbox)

    # Filter out small bounding boxes inside larger ones
    detected_bbox = filter_bboxes(detected_bbox)

    return detected_bbox


def crop_detected_section(image, bbox):
    """Crop the detected sections of the image based on bounding boxes."""
    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
    cropped_section = np.array(image)[y1:y2, x1:x2]
    return cropped_section


def apply_manga_ocr(section):
    """Applies MangaOcr to a section of an image.

    Args:
        section: A NumPy array representing the image section.

    Returns:
        The extracted text from the image section.
    """

    # Convert section to a compatible image format (e.g., BGR)
    if section.dtype != np.uint8:
        section = (section * 255).astype(np.uint8)

    if len(section.shape) == 2:
        image = cv2.cvtColor(section, cv2.COLOR_GRAY2BGR)
    elif section.shape[2] == 4:  # RGBA image (with alpha channel)
        image = cv2.cvtColor(section, cv2.COLOR_RGBA2BGR)
    else:
        image = section

    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(image)

    text = ocr(image)

    return text


def translate_japanese_to_english(text):
    """Translate Japanese text to English using MarianMT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def calculate_font_size(area, num_characters, min_font_size=5, max_font_size=100):
    """
    Calculates a dynamic font size based on the available area and number of characters.

    Args:
        area (int): The area of the rectangle where the text will be placed (width * height).
        num_characters (int): The number of characters in the text to be displayed.
        min_font_size (int, optional): The minimum allowed font size. Defaults to 10.
        max_font_size (int, optional): The maximum allowed font size. Defaults to 40.

    Returns:
        int: The calculated font size.
    """

    # Adjust area to account for character width and spacing (heuristic factor)
    adjusted_area = area / (num_characters * 1.2)

    # Calculate target font size based on square root of area
    target_font_size = math.sqrt(adjusted_area)

    # Clamp font size within the specified range
    return max(min_font_size, min(target_font_size, max_font_size))


def replace_rectangles_with_white(
    image, ocr_bboxes, texts, font_path="./NotoSansJP-Regular.ttf", font_size=40
):
    """
    Replace specified OCR-detected rectangular regions with white color in an image.

    Parameters:
    - image_path (str): Path to the input image.
    - ocr_bboxes (list of list): List of bounding boxes from OCR, where each box is a list of points.
    - texts (list of str): List of text labels to draw above each rectangle.
    - font_path (str): Path to the Japanese font file.
    - font_size (int): Font size for text labels.

    Returns:
    - Image object with specified rectangles replaced by white.
    """
    # Open the image
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # # Load the Japanese font with the specified size
    # try:
    #     font = ImageFont.truetype(font_path, font_size)
    # except IOError:
    #     print("Font not found. Using default font, which may not support Japanese characters.")
    #     font = ImageFont.load_default(size = font_size)  # Fallback to default if the font file isn't available

    # Convert OCR polygon bounding boxes to rectangles and draw them
    for bbox, text in zip(ocr_bboxes, texts):
        # Flatten and find min/max to convert polygon to bounding rectangle
        x_coords, y_coords = zip(*bbox)
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

        # Calculate area of the rectangle
        area = (x2 - x1) * (y2 - y1)

        # Calculate dynamic font size
        font_size = calculate_font_size(area, len(text))

        font = ImageFont.load_default(size=font_size)

        # Draw rectangle with white fill
        draw.rectangle([x1, y1, x2, y2], fill="white")

        # Draw bounding box outline (adjust line width as needed)
        # draw.rectangle([x1, y1, x2, y2], outline="black", width=2)

        # Wrap text based on rectangle width
        max_text_width = x2 - x1
        wrapped_text = textwrap.fill(text, width=int(max_text_width / font_size * 1.8))

        # Draw each line of wrapped text
        text_y_position = y1  # Start position for text
        for line in wrapped_text.splitlines():
            draw.text((x1, text_y_position), line, fill="black", font=font)
            text_y_position += font_size  # Move down for the next line

    return image


def main():

    st.set_page_config(
        page_title="GO_GO Nihongo Translator",  # Title of the tab
        page_icon="https://cdn3.iconfinder.com/data/icons/logos-brands-3/24/logo_brand_brands_logos_translate_google-512.png",  # You can use an emoji or a URL to an image
        layout="centered",  # Can be 'centered' or 'wide'
        initial_sidebar_state="expanded",  # Default state of the sidebar
    )

    st.title("GO_GO Nihongo Translator")
    st.write(
        "Upload an image to detect Japanese text, translate it to English, and replace the text in the image."
    )

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Create two columns for images
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Image:")
            display_image(image, "Uploaded Image")

        # Create overall status
        with st.status("Processing image...", expanded=True) as status:
            # Perform OCR
            # status.write("Detecting text...")
            status.update(label="Detecting text...", state="running")
            progress_bar = st.progress(0)
            detected_bbox = easyocr_detection(image)
            progress_bar.progress(25)

            # Extract text
            # status.write("Cropping Images ....")
            status.update(label="Cropping Images...", state="running")
            cropped_images = []
            for i, bbox in enumerate(detected_bbox):
                cropped_images.append(crop_detected_section(image, bbox))
                progress_bar.progress(25 + (i + 1) * 25 // len(detected_bbox))

            # Extract text
            # status.write("Extracting text with MangaOCR...")
            status.update(label="Extracting text with MangaOCR...", state="running")
            extracted_texts = [apply_manga_ocr(cropped) for cropped in cropped_images]

            # Translate text
            # status.write("Translating text to English...")
            status.update(label="Translating text to English...", state="running")
            translated_texts = []
            for i, text in enumerate(extracted_texts):
                translated_texts.append(translate_japanese_to_english(text))
                progress_bar.progress(50 + (i + 1) * 25 // len(extracted_texts))

            # Replace text
            # status.write("Replacing text in the image...")
            status.update(label="Replacing text in the image...", state="running")

            result_image = replace_rectangles_with_white(
                image, detected_bbox, translated_texts
            )
            progress_bar.progress(100)
            status.update(label="Processing complete!", state="complete")

        # # Perform OCR
        # st.write("Detecting text...")
        # detected_bbox = easyocr_detection(image)
        # cropped_images = [crop_detected_section(image, bbox) for bbox in detected_bbox]

        # # Extract and translate text
        # st.write("Extracting text with MangaOCR...")
        # extracted_texts = [apply_manga_ocr(cropped) for cropped in cropped_images]

        # st.write("Translating text to English...")
        # translated_texts = [
        #     translate_japanese_to_english(text) for text in extracted_texts
        # ]

        # # Replace text in the image
        # st.write("Replacing text in the image...")
        # result_image = replace_rectangles_with_white(
        #     image, detected_bbox, translated_texts
        # )

        # Display result image in second column
        with col2:
            st.write("Translated Image:")
            display_image(result_image, "Result Image")

        # Create table data
        table_data = {
            "Text #": range(1, len(extracted_texts) + 1),
            "Original Text": extracted_texts,
            "Translated Text": translated_texts,
        }

        # Display data as a table
        st.write("Detected and Translated Text:")
        st.table(pd.DataFrame(table_data))

        # Option to download the result
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        st.download_button(
            label="Download Result Image",
            data=buffer.getvalue(),
            file_name="translated_image.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
