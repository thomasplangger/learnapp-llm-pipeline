import io
from datetime import datetime
from fastapi import HTTPException, UploadFile
from pdfminer.high_level import extract_text
from app.db import fs, pdfs_metadata_coll, pdf_texts_coll


async def extract_pdf_text_and_store(file: UploadFile):
    """
    Store uploaded PDF into GridFS, extract simple metadata,
    and save metadata/document record.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDFs allowed")
    content = await file.read()

    grid_out = await fs.upload_from_stream(file.filename, io.BytesIO(content))
    pdf_id = str(grid_out)

    try:
        text = extract_text(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

    page_count = text.count('\f') + 1 if '\f' in text else text.count('\n\n') + 1
    word_count = len(text.split())

    await pdfs_metadata_coll.insert_one({
        "_id":         grid_out,
        "filename":    file.filename,
        "uploaded_at": datetime.utcnow(),
        "page_count":  page_count,
        "word_count":  word_count
    })

    await pdf_texts_coll.insert_one({
        "_id": grid_out,
        "text": text or "",
        "length": len(text or ""),
        "created_at": datetime.utcnow()
    })


    return {
        "pdf_id":     pdf_id,
        "filename":   file.filename,
        "page_count": page_count,
        "word_count": word_count,
    }
