import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inference import get_inference_service


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(
    title="Vertebra Landmark Detection API",
    version="0.1.0",
    description="Prototype API for vertebra landmark detection and Cobb angle estimation.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def frontend_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js", include_in_schema=False)
def frontend_app_js():
    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/styles.css", include_in_schema=False)
def frontend_styles_css():
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/health")
def health():
    service = get_inference_service()
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(service.device),
        "input_size": {
            "width": service.input_w,
            "height": service.input_h,
        },
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_overlay: bool = Query(True, description="Return a JPEG overlay as base64."),
):
    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    image_bytes = await file.read()

    try:
        service = get_inference_service()
        return service.predict(image_bytes=image_bytes, include_overlay=include_overlay)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
