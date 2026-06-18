import time
import cv2
import io
from PIL import Image

# library for both - google-genai
from google import genai
from google.genai import types

GEMINI_API_KEY = ""

def generate_image(input_video_path: str) -> Image.Image:
    """
    Extracts the first frame from a video file, passes it to the Gemini image
    model, saves a local debug copy, and directly returns the PIL Image object.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API Key is missing or not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Extract the first frame
    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to read the video file to extract a frame.")

    # Convert the frame from BGR to RGB, then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    prompt = (
        "Apply a semi-transparent, translucent color mask over the visible heart chambers and structures, "
        "targeting ONLY the black and dark pixel regions (the cavities). Do NOT paint over, outline, or obscure "
        "any white or bright tissue pixels—the white structures must remain completely untouched and clean. "
        "The color fills must be vibrant but transparent, blending like a tint over the dark background: "
        "Red mask for Left Ventricle (LV), Orange mask for Right Ventricle (RV), Cyan mask for Mitral Valve (MV), "
        "Magenta mask for Tricuspid Valve (TV), Green mask for Left Atrium (LA), and Yellow mask for Right Atrium (RA). "
        "Overlay small, static text abbreviations next to each masked region. Maintain the original image dimensions."
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=[prompt, pil_image],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE']
        )
    )

    if response.parts is None:
        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason.name
            raise Exception(f"Generation blocked by API. Finish reason: {reason}.")
        raise Exception("API returned an empty response. Likely blocked by safety filters.")

    for part in response.parts:
        if part.inline_data and part.inline_data.data:
            # Reconstruct a clean, native PIL Image straight from raw bytes
            clean_image = Image.open(io.BytesIO(part.inline_data.data))

            # (Optional) Keep your local debug save if you still want it
            clean_image.save("debug_gemini_output.png")

            return clean_image

    return None

def generate_medical_opinion(video_path: str) -> str:
    """
    Uploads the video to gemini-3.1-flash-lite for an expert medical evaluation.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API Key is missing or not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        video_file = client.files.upload(file=video_path)

        while True:
            video_file = client.files.get(name=video_file.name)
            if video_file.state.name == "PROCESSING":
                time.sleep(2)
            elif video_file.state.name == "FAILED":
                raise Exception("Gemini video processing failed on the server side.")
            else:
                break

        agent_prompt = """
        You are a Board-Certified Cardiologist and an elite specialist in Advanced Echocardiography Analysis. 
        Evaluate this uploaded Apical 4-Chamber (A4C) view echocardiogram video for clinical presentation data.

        Analyze systematically:
        1. Ventricular Performance
        2. Chamber Symmetrical Dimensions
        3. Valvular Kinematics
        4. Primary Findings & Concerns

        Provide your assessment using highly professional, clinical language. Conclude with a strong, mandatory medical disclaimer emphasizing that your analysis is an AI-generated assessment for educational demonstration and must be formally verified by a physician.
        """

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[video_file, agent_prompt]
        )

        return response.text

    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass