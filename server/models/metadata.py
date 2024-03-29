from pydantic import BaseModel, Field
from fastapi import UploadFile

class Metadata(BaseModel):
    id: str = Field(..., min_length=12, max_length=12, pattern="^[A-Za-z0-9]*$") # 12자리 random string
    age: int = Field(..., ge=10)
    gender: int = Field(..., ge=0, le=1) # 0: 여성, 1: 남성
    question: int = Field(..., ge=1, le=5) # 1: 행복한 기억, 2: 불행한 기억, 3: 긍정적 사진, 4: 중립적 사진, 5: 부정적 사진
    created_at: str = Field(...) # 생성 시간
    key: str = Field(...) # 통신을 위한 key
    audio_file: UploadFile = Field(...) # 음성 파일

    class Example:
        id = "AB3DCddk39Jl" # 12자리 random string
        age = 25
        gender = 1
        question = 1
        created_at = "2024-03-23 12:00:00"
        key = "d3d3LmNvbS8=" # env에 저장된 key값
        audio_file = "audio.wav"