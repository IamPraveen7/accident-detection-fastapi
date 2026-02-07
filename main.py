import os
import gc
import cv2
import shutil # upgrade
import datetime
import numpy as np
import tensorflow as tf
import uuid

from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
from functools import reduce
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ---------------- ENV ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
load_dotenv()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 100MB
INPUT_SIZE = (180, 180)
UPLOAD_FOLDER = "static/videos"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL ----------------
interpreter = tf.lite.Interpreter(model_path="accident_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- FASTAPI ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Detect route
@app.post("/detect", response_class=HTMLResponse)
async def detect(
    request: Request,
    bgtasks: BackgroundTasks,
    photage: UploadFile = File(...),
    recipients: str = Form(...),
    latitude: str = Form(None),
    longitude: str = Form(None),
):
    # --- per-request folder ---
    request_id = str(uuid.uuid4())
    request_dir = os.path.join(UPLOAD_FOLDER, request_id)
    os.makedirs(request_dir, exist_ok=True)
    
    video_path = os.path.join(request_dir, photage.filename)

    # --- file size validation ---
    size = 0
    with open(video_path, "wb") as f:
        while chunk := await photage.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                shutil.rmtree(request_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail="File too large (max 100MB)"
                )
            f.write(chunk)

    email_list = [e.strip() for e in recipients.split(",") if e.strip()]

    # --- background processing ---
    bgtasks.add_task(
        process_video_background,
        request_dir,
        video_path,
        email_list,
        latitude,
        longitude,
    )

    output = "⏳ Processing video… Alert will be sent if an accident is detected."

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file_path": video_path,
            "out": output,
        },
    )

# bg task ml+email+cleanup
def process_video_background(video_dir, video_path, recipients, latitude, longitude):
    try:
        predictions = preprocess_video(video_path)
        if not predictions:
            return

        avg_pred = float(np.mean(predictions))
        if avg_pred > 0.5:
            send_alerts(avg_pred, recipients, latitude, longitude)

    except Exception as e:
        print("Background processing error:", e)

    finally:
        shutil.rmtree(video_dir, ignore_errors=True)
        gc.collect()

# # Predict Accident
# def predict_accident(video_path, recipients, latitude, longitude):
#     predictions = preprocess_video(video_path)
#     if not isinstance(predictions, list):
#         return predictions

#     avg_pred = np.mean(predictions)
#     label = "🚨 Accident Detected" if avg_pred > 0.5 else "✅ No Accident"

#     if label == "🚨 Accident Detected":
#         send_alerts(avg_pred, recipients, latitude, longitude)

#     return {label: float(avg_pred)}

# Preprocess the given video
def preprocess_video(video_path):
    predictions = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return predictions

    sec = 0
    while sec < 10:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, INPUT_SIZE)
            frame = frame.astype("float32") / 255.0
            frame = frame[np.newaxis, ...]

            interpreter.set_tensor(input_details[0]["index"], frame)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]["index"])[0][0]

            predictions.append(prob)

            if prob > 0.9:
                break

            sec += 2
        except Exception as e:
            print("Frame skipped: ",e)
            sec+=2
            continue

    cap.release()
    return predictions

# trigger emails
def send_alerts(avg_pred, recipients, lt, lg):
    subject = f"🚨 Accident Detected | {datetime.datetime.now().strftime('%H:%M:%S')}"

    map_link = ""

    if lt and lg:
        map_link = f"https://www.google.com/maps?q={lt},{lg}"

    dt = list(datetime.datetime.now().ctime().split(" "))
    if dt[2] == "":
        time = dt.pop(4)
        date = reduce(lambda x,y : x + " " + y, dt)
    else:
        time = dt.pop(3)
        date = reduce(lambda x,y : x + " " + y, dt)

    html_body = f"""
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
    message = Mail(
        from_email=os.getenv('SENDGRID_SENDER'),
        to_emails=recipients,
        subject=subject,
        html_content=html_body)
    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sg.send(message)
    except Exception as e:
        print("sendgrid Error: ",e)

