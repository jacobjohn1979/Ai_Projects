from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("test_files", exist_ok=True)

def create_image(filename, text):
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    draw.text((50, 150), text, fill="black")

    img.save(f"test_files/{filename}")

# Clean image
create_image("clean_image.jpg", "John Doe | Amount: 1200.00 | Date: 10/03/2026")

# Tampered image
create_image("tampered_image.jpg", "John Doe | Amount: 9800.00 | Date: 10/03/2026")

# Extreme fraud case
create_image("fake_high_value.jpg", "John Doe | Amount: 99999.00 | Date: 01/01/2020")

print("Image test files created in /test_files")