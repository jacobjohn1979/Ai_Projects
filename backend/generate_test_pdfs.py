from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

os.makedirs("test_files", exist_ok=True)

def create_pdf(filename, text):
    c = canvas.Canvas(f"test_files/{filename}", pagesize=letter)
    c.drawString(100, 700, text)
    c.save()

# Clean PDF
create_pdf("clean_statement.pdf", "Account: John Doe | Amount: 1,200.00 | Date: 10/03/2026")

# Tampered PDF (simulated edit)
create_pdf("tampered_statement.pdf", "Account: John Doe | Amount: 9,800.00 | Date: 10/03/2026")

# Another variation
create_pdf("suspicious_statement.pdf", "Account: John Doe | Amount: 99,999.00 | Date: 01/01/2020")

print("PDF test files created in /test_files")