import os
import uuid
import shutil
import datetime
from functools import reduce
import json

import cv2
import numpy as np
# ---------------- ENV ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
UPLOAD_FOLDER = "static/videos"
INPUT_SIZE = (180, 180)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL ----------------
interpreter = tf.lite.Interpreter(model_path="accident_detection_model.tflite")

# ---------------- FASTAPI ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect", response_class=HTMLResponse)
async def detect(
    request: Request,
    photage: UploadFile = File(...)
    ):
    # ---------- per-request folder ----------
    request_id = str(uuid.uuid4())
    request_dir = os.path.join(UPLOAD_FOLDER, request_id)
    os.makedirs(request_dir, exist_ok=True)

    if not photage.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Invalid video format")
    video_path = os.path.join(request_dir, photage.filename)

    # ---------- file size validation ----------
    size = 0
    with open(video_path, "wb") as f:
        while True:
            chunk = await photage.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                shutil.rmtree(request_dir, ignore_errors=True)
                raise HTTPException(413, "File too large (max 25MB)")
            f.write(chunk)

    # ---------- background ML ----------
    # Run ML synchronously (intentional)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    preds = preprocess_video(video_path, input_details, output_details)
    avg = float(np.mean(preds)) if preds else 0.0
    label = f"🚨 Accident Detected" if avg > 0.5 else "✅ No Accident"
    output = f"🚨 Accident Detected with {avg*100:.1f}% chance" if label.startswith("🚨") else "✅ No Accident"

    # Save result
    with open(os.path.join(request_dir, "result.json"), "w") as f:
        json.dump(
            {"avg": avg, "label": label, "video": video_path},
            f
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file_path": video_path,
            "out": output,
            "show_email" : label.startswith("🚨"),
            "request_id": request_id,   # ✅
        },
    )
# synchronous
def preprocess_video(video_path, input_details, output_details):
    preds = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return preds
    
    sec = 0
    while sec < 10:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, INPUT_SIZE)
            frame = frame.astype("float32") / 255.0
            frame = frame[np.newaxis, ...]

            interpreter.set_tensor(input_details[0]["index"], frame)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]["index"])[0][0]

            preds.append(prob)

            if prob > 0.9:
                break

            sec += 2
        except Exception:
            sec += 2

    cap.release()
    return preds


#send-alert
@app.post("/send-email")
def send_alert_only(
    request : Request,
    request_id: str = Form(...),
    recipients: str = Form(...),
    latitude: str | None = Form(None),
    longitude: str | None = Form(None),
    ):
    result_file = os.path.join(UPLOAD_FOLDER, request_id, "result.json")
    if not os.path.exists(result_file):
        raise HTTPException(404, "Result not found")

    with open(result_file) as f:
            data = json.load(f)
    try:
        send_alert(
        data["avg"],
        [e.strip() for e in recipients.split(",") if e.strip()],
        latitude,
        longitude,
        )

        return templates.TemplateResponse(
            "status.html",{
                "request": request,
                "recipients": recipients,
            },
        )
    except Exception as e:
        print("email exception", e)

def send_alert(avg_pred, recipients, lat, lng):
    subject = f"🚨 Accident Detected | {datetime.datetime.now():%H:%M:%S}"

    map_link = (
        f"https://www.google.com/maps?q={lat},{lng}"
        if lat and lng
        else "Location unavailable"
    )
    dt = list(datetime.datetime.now().ctime().split(" "))
    if dt[2] == "":
        time = dt.pop(4)
        date = reduce(lambda x,y : x + " " + y, dt)
    else:
        time = dt.pop(3)
        date = reduce(lambda x,y : x + " " + y, dt)

    html = f"""
    <html>
        <body style="font-family: Arial, sans-serif;">
        <h2 style="color:red;">An accident has been detected</h2>
        <p><strong>📊 Probability:</strong> {avg_pred * 100:.1f}%</p>
        <p><strong>⏱️ Time:</strong> {time.strip()}</p>
        <p><strong>📅 Date:</strong> {date.strip()}</p>
        <p><strong>📍 Location:</strong><a href="{map_link}" target="_blank"> Google Maps Link</a></p>
        <hr style="border:none;height:1px;background:#ddd;">
        <h3 style="color:red;"><strong>Please respond immediately.</strong></h3>
    </body>
    </html>
    """
    api_key = os.getenv("SENDGRID_API_KEY")
    sender = os.getenv("SENDGRID_SENDER")

    if not api_key or not sender:
        raise RuntimeError("SENDGRID_API_KEY or SENDGRID_SENDER not set")

    message = Mail(
        from_email=sender,
        to_emails=recipients,
        subject=subject,
        html_content=html,
    )

    sg = SendGridAPIClient(api_key)
    response = sg.send(message)
    print(response.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
