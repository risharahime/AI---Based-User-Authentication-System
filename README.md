# AI-Based User Authentication and Identity Card Verification System

## Project Overview

This project is an AI-powered user authentication and identity verification system developed using Python and Streamlit.

It combines facial recognition and Optical Character Recognition (OCR) to authenticate users and verify identity card details.

The system ensures that users must successfully pass live facial authentication before proceeding to identity card verification.

## Key Features

- **User Registration:** Register users with their name, date of birth, and facial image.
- **Live Face Authentication:** Verify users by comparing their live captured face with their registered facial image.
- **Identity Card Verification:** Allow users to upload an identity card only after successful facial authentication.
- **Face Matching:** Compare the photograph on the identity card with the registered facial image.
- **OCR-Based Text Extraction:** Extract information from identity cards using Tesseract OCR.
- **Name Verification:** Compare the extracted name with the registered user name.
- **Date of Birth Verification:** Verify the extracted date of birth against the registered details.
- **Final Verification Results:** Display the results of the different authentication and verification stages.

## Technologies Used

- Python
- Streamlit
- DeepFace
- OpenCV
- NumPy
- Tesseract OCR
- Pytesseract
- SQLite
- Pillow

## Project Workflow

1. User registers by providing personal details and capturing a facial image.
2. The system stores the registration information.
3. The user performs live facial authentication.
4. If authentication is successful, the user can upload an identity card.
5. The system compares the identity card photograph with the registered face.
6. OCR extracts the name and date of birth from the identity card.
7. The system compares the extracted details with the registered information.
8. Final verification results are displayed.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd AI_User_Authentication
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

## Important Note

Tesseract OCR must be installed separately on the system. The application also requires the appropriate Tesseract executable path configuration.

## Project Purpose

The main objective of this project is to develop a multi-stage identity verification system that combines artificial intelligence, facial recognition, and OCR to improve the user authentication process.
