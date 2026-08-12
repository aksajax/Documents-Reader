import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# 1. Hindi PaddleOCR Initialization
# lang='hi' sets Hindi/Devanagari model
ocr = PaddleOCR(use_angle_cls=True, lang='hi', show_log=False)

def process_hindi_legal_pdf(pdf_path, dpi=300):
    print(f"Converting PDF to Images: {pdf_path}")
    
    # PDF ke sabhi pages ko images me convert karein
    pages = convert_from_path(pdf_path, dpi=dpi)
    full_document_text = []

    for page_num, page in enumerate(pages):
        print(f"Processing Page {page_num + 1}/{len(pages)}...")
        
        # PIL Image ko OpenCV numpy array me convert karein
        image_np = np.array(page)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Preprocessing: Grayscale & Contrast Enhance
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Temporary image save for OCR input
        temp_img_path = f"temp_page_{page_num}.png"
        cv2.imwrite(temp_img_path, gray)

        # 2. PaddleOCR Run Karein
        result = ocr.ocr(temp_img_path, cls=True)

        page_text = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]  # Extracted Text
                confidence = line[1][1]  # OCR Confidence score
                if confidence > 0.4:  # Low confidence noise filter
                    page_text.append(text)

        # Clean temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        page_content = f"\n--- Page {page_num + 1} ---\n" + "\n".join(page_text)
        full_document_text.append(page_content)

    return "\n".join(full_document_text)

# Example Usage
# extracted_text = process_hindi_legal_pdf("sample_sale_deed.pdf")
# print(extracted_text[:1000])  # Pehle 1000 characters dekhein 