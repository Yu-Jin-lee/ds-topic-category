import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

app = FastAPI()

# 데이터 로드 (JSON 파일을 서버 시작 시 미리 불러올 수 있습니다)
with open('/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict.json', 'r', encoding='utf-8') as f:
    depth_dict = json.load(f)

@app.get("/categories")
def get_categories():
    return JSONResponse(content=depth_dict)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
