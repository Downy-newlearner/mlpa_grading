from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = FastAPI()

@app.get("/pdf/course-stats")
def course_stats_pdf(subject: str = Query(default="TEST_SUBJECT")):
    mean_score = 87.12
    std_dev = 5.34

    pdf_buf = io.BytesIO()

    c = canvas.Canvas(pdf_buf, pagesize=letter)
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, f"Subject: {subject}")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Mean: {mean_score:.2f}, Std: {std_dev:.2f}")
    c.showPage()
    c.save()

    pdf_buf.seek(0)

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=course_report.pdf"
        }
    )

from pydantic import BaseModel
from typing import List, Optional

class QuestionModel(BaseModel):
    questionId: Optional[int] = None
    examCode: Optional[str] = None
    questionNumber: int
    questionType: Optional[str] = None
    subQuestionNumber: Optional[int] = None
    answer: Optional[str] = None
    answerCount: int
    point: float

@app.post("/grading/questions")
def receive_questions(questions: List[QuestionModel]):
    print(f"Received {len(questions)} questions")
    for q in questions:
        print(q)
    return {"status": "success", "count": len(questions)}
