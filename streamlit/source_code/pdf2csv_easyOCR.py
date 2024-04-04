import os
from pathlib import Path
import cv2
from pdf2image import convert_from_path
import numpy as np
import easyocr

def ocr_result(pdf_path):
    pages = convert_from_path(pdf_path)
    reader = easyocr.Reader(['en'], gpu=False)

    # Loop through each page
    for idx, page in enumerate(pages):
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        # Sharpen the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

        # Thresholding
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform OCR
        ocr_results = reader.readtext(thresh, detail=0)

        # Print OCR results
        print(f"Page {idx + 1} OCR Result:\n{ocr_results}\n")

if __name__ == "__main__":
    print(os.getcwd())
    data_folder = Path(os.getcwd()).parent / 'Financial_Diaries'
    glob_pattern = '**/*.pdf'
    filepath_gen = data_folder.glob(glob_pattern)

    for file in filepath_gen:
        ocr_result(str(file))