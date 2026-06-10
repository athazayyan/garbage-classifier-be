from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import numpy as np
import onnxruntime as ort

from PIL import Image
from io import BytesIO

app = FastAPI(
    title="Garbage Classification API"
)

# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Load ONNX Model
# ==========================

session = ort.InferenceSession(
    "model/mobilenet.onnx"
)

INPUT_NAME = session.get_inputs()[0].name

CLASS_NAMES = [
    "glass",
    "metal",
    "paper",
    "plastic"
]

# ==========================
# Image Preprocessing
# ==========================

def preprocess_image(contents):

    image = Image.open(
        BytesIO(contents)
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    # Sama seperti training
    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================
# Routes
# ==========================

@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/classes")
def classes():

    return {
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = preprocess_image(
        contents
    )

    prediction = session.run(
        None,
        {
            INPUT_NAME: image
        }
    )

    probabilities = prediction[0][0]

    idx = int(
        np.argmax(probabilities)
    )

    label = CLASS_NAMES[idx]

    confidence = float(
        probabilities[idx]
    )

    return {
        "label": label,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": {
            CLASS_NAMES[i]:
            round(
                float(probabilities[i]) * 100,
                2
            )
            for i in range(
                len(CLASS_NAMES)
            )
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
