import os
from pathlib import Path

print(os.getcwd())
data_folder = Path(os.getcwd()).parent / 'Financial_Diaries'
glob_pattern = '**/*.pdf'
filepath_gen = data_folder.glob(glob_pattern)

pdf_files = [file for file in list(filepath_gen) if file.is_file() and file.suffix.lower() == '.pdf']

for pdf_file in pdf_files:
    print(pdf_file)

print(f'Total PDF files: {len(pdf_files)}')

#%%
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2


def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def ocr_handwritten_text(image):
    text = pytesseract.image_to_string(image, lang='eng')
    return text


def main(pdf_path):
    images = pdf_to_images(pdf_path)

    for idx, image in enumerate(images):
        text = ocr_handwritten_text(image)
        print(f"Page {idx + 1} OCR Result:\n{text}\n")


# if __name__ == "__main__":
#     for pdf_file in pdf_files:
#     main(pdf_file)

# pdf_path = "/Users/jiwoosuh/Desktop/JIWOO/SP24/Capstone_WB/Financial_Diaries/Baseline Financial Diaries 2/Ogun/FDs/NFWP Ogun State Financial Diary 3rd Recall.pdf"
pdf_path = "/Users/jiwoosuh/Desktop/JIWOO/SP24/Capstone_WB/Financial_Diaries/Baseline Financial Diaries 2/Kebbi/FDs/Ngaski/WAG/kebbi_ngaski_wara_FD_WAG_10122021. week 1.pdf"

if __name__ == "__main__":
    main(pdf_path)

