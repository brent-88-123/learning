import sys
import os
import comtypes.client

def doc_to_pdf(doc_path):
    try:
        pdf_path = os.path.splitext(doc_path)[0] + ".pdf"

        word = comtypes.client.CreateObject("Word.Application")
        word.Visible = False
        
        try:
            doc = word.Documents.Open(doc_path)
            doc.SaveAs(pdf_path, FileFormat=17)  # 17 = wdFormatPDF
            doc.Close()
        finally:
            word.Quit()

        return pdf_path
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_pdf.py <path_to_doc>", file=sys.stderr)
        sys.exit(1)

    doc_path = sys.argv[1]
    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}", file=sys.stderr)
        sys.exit(1)

    pdf_path = doc_to_pdf(doc_path)
    print(f"PDF saved to: {pdf_path}")