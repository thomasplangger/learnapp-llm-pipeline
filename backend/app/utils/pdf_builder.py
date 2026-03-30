import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from fastapi.responses import StreamingResponse

def build_pdf_from_text(guide_text: str, filename: str) -> StreamingResponse:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40
    p.setFont("Helvetica-Bold", 16)
    p.drawString(40, y, filename)
    y -= 30

    p.setFont("Helvetica", 11)
    for line in guide_text.split("\n"):
        if y < 60:
            p.showPage()
            y = height - 40
            p.setFont("Helvetica", 11)
        p.drawString(40, y, line)
        y -= 14

    p.showPage()
    p.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
    )
