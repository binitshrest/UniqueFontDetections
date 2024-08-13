import pytesseract
from PIL import Image
import cv2
import os
from font_detection import detect_font_features


def extract_text(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to extract text from the image
    filename = "{}.jpeg".format(os.getpid())

    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(gray)
    return text


def find_hidden_word(text, font_labels):
    words = text.split()
    hidden_word = ""
    word_font_map = {}

    font_labels = font_labels.tolist()

    for i, word in enumerate(words):
        word_length = len(word)
        if 1 <= word_length < 7 and 'o' in word:
            # Check if the word's font is unique
            if font_labels.count(font_labels[i]) == 1:
                hidden_word = word
                break

    return hidden_word


# Example of processing multiple image files
image_files = ['image1.jpeg', 'image2.jpeg', "image3.jpeg", "image4.jpeg", "image5.jpeg","image6.jpeg","image7.jpeg","image8.jpeg","image9.jpeg","image10.jpeg","image11.jpeg","image12.jpeg","image13.jpeg","image14.jpeg","image15.jpeg","image16.jpeg", "image17.jpeg","image18.jpeg"]  # Add your image files here

for image_file in image_files:
    text = extract_text(image_file)
    font_labels = detect_font_features(image_file)
    hidden_word = find_hidden_word(text, font_labels)
    print(f"Hidden word in {image_file}: {hidden_word}")
