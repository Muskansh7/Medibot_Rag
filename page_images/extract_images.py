import fitz
import os

pdf_path = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

output_folder = "page_images"

os.makedirs(output_folder, exist_ok=True)

doc = fitz.open(pdf_path)

for page_num in range(len(doc)):

    page = doc[page_num]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    image_path = f"{output_folder}/page_{page_num+1}.png"

    pix.save(image_path)

print("Page images extracted successfully")