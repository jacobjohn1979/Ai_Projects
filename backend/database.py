import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ScreeningLog(Base):
    __tablename__ = "screening_logs"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(Text)
    file_type = Column(Text)
    screening_type = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    risk_score = Column(Integer)
    risk_level = Column(Text)
    status = Column(Text)
    engine_version = Column(Text)
    flags = Column(JSONB)
    metadata_json = Column("metadata", JSONB)
    # metadata = Column(JSONB)
    ocr_summary = Column(JSONB)
    forensics = Column(JSONB)
    field_checks = Column(JSONB)
    raw_result = Column(JSONB)


Base.metadata.create_all(bind=engine)


def save_screening_log(result, file_name, file_type, screening_type):
    db = SessionLocal()
    try:
        log = ScreeningLog(
            file_name=file_name,
            file_type=file_type,
            screening_type=screening_type,
            processed_at=datetime.utcnow(),
            risk_score=result.get("risk_score"),
            risk_level=result.get("risk_level"),
            status="completed",
            engine_version="v1",
            flags=result.get("flags", []),
            metadata_json=result.get("metadata") or result.get("forensics"),
            ocr_summary=result.get("ocr_summary") or result.get("ocr"),
            forensics=result.get("image_forensics") or result.get("forensics"),
            field_checks=result.get("field_checks"),
            raw_result=result,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()