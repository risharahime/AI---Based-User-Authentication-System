
import os
os.environ["DEEPFACE_BACKEND_ENGINE"] = "onnx"

import re
import sqlite3
import hashlib
from io import BytesIO
from pathlib import Path
from datetime import date

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from deepface import DeepFace

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI User Authentication",
    page_icon="🔐",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------- DATABASE ----------------
# Uses the existing users.db. It does NOT delete or reset registrations.

conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dob TEXT NOT NULL,
    face_image BLOB NOT NULL
)
""")
conn.commit()

# ---------------- FACE FUNCTIONS ----------------

@st.cache_resource
def load_facenet_model():
    return DeepFace.build_model("Facenet")


def image_bytes_to_bgr(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def uploaded_file_to_bgr(uploaded_file):
    return image_bytes_to_bgr(uploaded_file.getvalue())


def verify_two_faces(image1_bgr, image2_bgr):
    result = DeepFace.verify(
        img1_path=image1_bgr,
        img2_path=image2_bgr,
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="cosine",
        enforce_detection=True,
        align=True,
        silent=True
    )
    return bool(result["verified"]), float(result["distance"])


def extract_id_card_face(card_bgr):
    faces = DeepFace.extract_faces(
        img_path=card_bgr,
        detector_backend="opencv",
        enforce_detection=True,
        align=True
    )

    if len(faces) != 1:
        raise ValueError(
            f"Expected one photograph on the ID card, but detected {len(faces)} faces."
        )

    face_rgb = faces[0]["face"]
    if face_rgb.dtype != np.uint8:
        if face_rgb.max() <= 1.0:
            face_rgb = face_rgb * 255.0
        face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)

    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)


# ---------------- OCR FUNCTIONS ----------------

def extract_details(ocr_text):
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    date_pattern = (
        r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|"
        r"\d{4}[/-]\d{2}[/-]\d{2})\b"
    )

    extracted_dob = ""
    for value in re.findall(date_pattern, ocr_text):
        try:
            if re.match(r"^\d{4}", value):
                parsed = date.fromisoformat(value.replace("/", "-"))
            else:
                parsed = date(
                    int(value[6:10]),
                    int(value[3:5]),
                    int(value[0:2])
                )
            extracted_dob = parsed.isoformat()
            break
        except ValueError:
            continue

    extracted_name = ""
    for line in lines:
        match = re.search(
            r"\b(?:name|naam)\b\s*[:\-]?\s*(.*)",
            line,
            flags=re.IGNORECASE
        )
        if match:
            value = re.split(
                r"\b(?:dob|date of birth|birth)\b",
                match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE
            )[0]
            value = re.sub(r"[^A-Za-z .'-]", " ", value)
            extracted_name = " ".join(value.split()).strip()
            if extracted_name:
                break

    if not extracted_name:
        for index, line in enumerate(lines):
            if re.search(r"\b(?:name|naam)\b", line, re.IGNORECASE):
                if index + 1 < len(lines):
                    candidate = re.sub(
                        r"[^A-Za-z .'-]", " ", lines[index + 1]
                    )
                    candidate = " ".join(candidate.split()).strip()
                    if candidate and not re.search(
                        r"government|india|dob|birth|address|identity|card|uidai",
                        candidate,
                        re.IGNORECASE
                    ):
                        extracted_name = candidate
                break

    return extracted_name, extracted_dob


def normalize_name(name):
    return " ".join(name.lower().strip().split())


# ---------------- APPLICATION ----------------

st.title("🔐 AI-Based User Authentication System")
st.subheader("Using Face Recognition and Identity Card Verification")
st.write(
    "The system checks the live face first. Only a recognized registered "
    "user can continue to identity card verification."
)

try:
    load_facenet_model()
    model_ready = True
except Exception as error:
    model_ready = False
    st.error("FaceNet model could not be loaded.")
    st.code(str(error))

register_tab, verify_tab = st.tabs(
    ["User Registration", "User Authentication"]
)

# ---------------- USER REGISTRATION ----------------

with register_tab:
    st.header("User Registration")

    name = st.text_input("Enter Full Name")
    dob = st.date_input(
        "Enter Date of Birth",
        value=date(2000, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today()
    )

    st.subheader("Capture Face Image")
    face_upload = st.camera_input(
        "Take a clear photo of your face",
        key="register_camera"
    )

    if st.button("Register User"):
        if not model_ready:
            st.error("FaceNet model is not ready.")
        elif not name.strip():
            st.error("Please enter your name.")
        elif face_upload is None:
            st.error("Please capture your face image.")
        else:
            try:
                registration_bgr = uploaded_file_to_bgr(face_upload)
                DeepFace.extract_faces(
                    img_path=registration_bgr,
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=True
                )

                cursor.execute(
                    """
                    SELECT id FROM users
                    WHERE LOWER(name)=LOWER(?) AND dob=?
                    """,
                    (name.strip(), dob.isoformat())
                )
                existing = cursor.fetchone()

                if existing:
                    st.warning(
                        "This name and date of birth are already registered. "
                        "The existing record was not changed."
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO users (name, dob, face_image)
                        VALUES (?, ?, ?)
                        """,
                        (name.strip(), dob.isoformat(), face_upload.getvalue())
                    )
                    conn.commit()
                    st.success("User Registered Successfully!")

            except Exception as error:
                st.error("Registration could not be completed.")
                st.code(str(error))


# ---------------- USER AUTHENTICATION ----------------

with verify_tab:
    st.header("User Authentication")

    st.subheader("Step 1: Live Face Capture")
    login_face = st.camera_input(
        "Capture your face for verification",
        key="login_camera"
    )

    current_capture_hash = None
    if login_face is not None:
        current_capture_hash = hashlib.sha256(
            login_face.getvalue()
        ).hexdigest()

    # If the user takes a different photo, require face verification again.
    if (
        st.session_state.get("verified_capture_hash")
        and current_capture_hash != st.session_state.get("verified_capture_hash")
    ):
        st.session_state.pop("verified_user_id", None)
        st.session_state.pop("verified_capture_hash", None)

    if st.button("Verify Live Face"):
        if not model_ready:
            st.error("FaceNet model is not ready. Check the error above.")
        elif login_face is None:
            st.error("Please capture your face first.")
        else:
            cursor.execute(
                "SELECT id, name, dob, face_image FROM users ORDER BY id"
            )
            users = cursor.fetchall()

            if not users:
                st.error("No registered users found. Please register first.")
            else:
                try:
                    live_face_bgr = uploaded_file_to_bgr(login_face)
                    matched_user = None
                    best_distance = float("inf")

                    with st.spinner("Checking your face against registered users..."):
                        for user in users:
                            stored_face_bgr = image_bytes_to_bgr(user[3])
                            is_match, distance = verify_two_faces(
                                live_face_bgr,
                                stored_face_bgr
                            )
                            if is_match and distance < best_distance:
                                matched_user = user
                                best_distance = distance

                    if matched_user is not None:
                        st.session_state["verified_user_id"] = matched_user[0]
                        st.session_state["verified_capture_hash"] = current_capture_hash
                        st.success(
                            f"Live face matched the registered user: {matched_user[1]}."
                        )
                        st.info(
                            "Face verification passed. You can now upload the identity card."
                        )
                    else:
                        st.session_state.pop("verified_user_id", None)
                        st.session_state.pop("verified_capture_hash", None)
                        st.error(
                            "Face not recognized. Authentication rejected. "
                            "Identity card upload is not available."
                        )

                except Exception as error:
                    st.session_state.pop("verified_user_id", None)
                    st.session_state.pop("verified_capture_hash", None)
                    st.error("Live face verification could not be completed.")
                    st.code(str(error))

    # The ID card uploader is displayed ONLY after successful live face matching.
    verified_user_id = st.session_state.get("verified_user_id")
    verified_capture_hash = st.session_state.get("verified_capture_hash")

    if (
        verified_user_id is not None
        and current_capture_hash is not None
        and current_capture_hash == verified_capture_hash
    ):
        cursor.execute(
            "SELECT id, name, dob, face_image FROM users WHERE id=?",
            (verified_user_id,)
        )
        matched_user = cursor.fetchone()

        if matched_user is not None:
            st.divider()
            st.subheader("Step 2: Upload Identity Card")
            st.caption(
                "Your live face has been recognized. Now upload the identity "
                "card containing the person's photograph, name and DOB."
            )

            id_card = st.file_uploader(
                "Upload Identity Card",
                type=["jpg", "jpeg", "png"],
                key="identity_card_upload"
            )

            if st.button("Verify Identity Card"):
                if id_card is None:
                    st.error("Please upload your identity card.")
                else:
                    try:
                        card_bgr = uploaded_file_to_bgr(id_card)

                        st.subheader("Step 3: ID Card Photograph Verification")
                        card_face_bgr = extract_id_card_face(card_bgr)

                        registered_face_bgr = image_bytes_to_bgr(matched_user[3])
                        id_photo_match, _ = verify_two_faces(
                            card_face_bgr,
                            registered_face_bgr
                        )

                        st.subheader("Step 4: Identity Card OCR")
                        extracted_text = pytesseract.image_to_string(
                            Image.open(BytesIO(id_card.getvalue())).convert("RGB")
                        )
                        extracted_name, extracted_dob = extract_details(extracted_text)

                        st.write("Extracted Text:")
                        st.text(extracted_text)
                        st.write("Extracted Name:", extracted_name or "Not Found")
                        st.write("Extracted DOB:", extracted_dob or "Not Found")

                        name_match = (
                            bool(extracted_name)
                            and normalize_name(extracted_name)
                            == normalize_name(matched_user[1])
                        )
                        dob_match = (
                            bool(extracted_dob)
                            and extracted_dob == matched_user[2]
                        )

                        st.subheader("Final Verification Results")
                        st.write("Live Face vs Registered Face: ✅ Passed")
                        st.write(
                            "ID Card Photo vs Registered Face:",
                            "✅ Passed" if id_photo_match else "❌ Failed"
                        )
                        st.write(
                            "Name Verification:",
                            "✅ Passed" if name_match else "❌ Failed"
                        )
                        st.write(
                            "DOB Verification:",
                            "✅ Passed" if dob_match else "❌ Failed"
                        )

                        if id_photo_match and name_match and dob_match:
                            st.success(
                                "🎉 AUTHENTICATION SUCCESSFUL! USER VERIFIED."
                            )
                            st.balloons()
                            st.write("Welcome,", matched_user[1])
                        else:
                            st.error(
                                "❌ AUTHENTICATION REJECTED! "
                                "One or more identity card checks failed."
                            )

                    except Exception as error:
                        st.error(
                            "Identity card verification could not be completed. "
                            "Check that the card has one clear face photograph "
                            "and readable name and DOB."
                        )
                        st.code(str(error))

            if st.button("Start Over"):
                st.session_state.pop("verified_user_id", None)
                st.session_state.pop("verified_capture_hash", None)
                st.rerun()


# ---------------- REGISTERED USERS ----------------

st.divider()
st.header("Registered Users")

cursor.execute("SELECT id, name, dob FROM users ORDER BY id")
registered_users = cursor.fetchall()

if registered_users:
    st.dataframe(
        [
            {
                "User ID": user[0],
                "Name": user[1],
                "Date of Birth": user[2]
            }
            for user in registered_users
        ],
        use_container_width=True
    )
else:
    st.info("No registered users available.")

conn.close()
