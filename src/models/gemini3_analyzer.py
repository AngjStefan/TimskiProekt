import os
import time
import cv2
import io
from PIL import Image

from dotenv import load_dotenv
# library for both - google-genai
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

def generate_image(input_video_path: str, model_name: str) -> Image.Image:
    """
    Extracts the first frame from a video file, passes it to the specified Gemini image
    model with strict isolation guidelines, and returns a clean PIL Image object.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to read the video file to extract a frame.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Tightened prompt to eliminate segment crowding/hallucinations
    prompt = (
        "Apply a semi-transparent, translucent color mask over ONLY THE VISIBLE heart chambers and structures, "
        "targeting ONLY the black and dark pixel regions (the cavities). Do NOT paint over, outline, or obscure "
        "any white or bright tissue pixels—the white structures must remain completely untouched and clean. "
        "The color fills must be vibrant but transparent, blending like a tint over the dark background: "
        "Red mask for Left Ventricle (LV), Orange mask for Right Ventricle (RV), Cyan mask for Mitral Valve (MV), "
        "Magenta mask for Tricuspid Valve (TV), Green mask for Left Atrium (LA), and Yellow mask for Right Atrium (RA). "
        "If any of these 6 structures are not clearly visible or are partially obscured, only mask the clearly visible sections and leave the rest unmasked. "
        "Overlay small, static text abbreviations next to each masked region. Maintain the original image dimensions."
    )

    config_params = {"response_modalities": ['IMAGE']}
    if "pro" in model_name.lower() or "preview" in model_name.lower():
        config_params["thinking_config"] = types.ThinkingConfig(thinking_level="High")

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, pil_image],
        config=types.GenerateContentConfig(**config_params)
    )

    if response.parts is None:
        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason.name
            raise Exception(f"Generation blocked by API. Finish reason: {reason}.")
        raise Exception("API returned an empty response. Likely blocked by safety filters.")

    for part in response.parts:
        if part.inline_data and part.inline_data.data:
            clean_image = Image.open(io.BytesIO(part.inline_data.data))
            clean_image.save("debug_gemini_output.png")
            return clean_image

    return None


def generate_medical_opinion(video_path: str, model_name: str) -> str:
    """
    Uploads the video to the chosen Gemini text/multimodal model for an evaluation
    anchored purely on clearly visible perspectives.
    """
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
        Evaluate this uploaded echocardiogram video for clinical presentation data.

        CRITICAL CLINICAL BOUNDARY:
        Base your findings, assumptions, and conclusions STRICTLY on what is explicitly visible in the footage. 
        If a structural region, specific chamber, wall, or valve system is obscured, poorly resolved, or entirely 
        outside the visual field, you must explicitly document that specific section as 'Visualization insufficient for clinical evaluation' 
        or state that no diagnostic inference can be made for that area due to missing visualization. Do not speculate or extrapolate.

        Analyze systematically:
        1. Ventricular Performance
        2. Chamber Symmetrical Dimensions
        3. Valvular Kinematics
        4. Primary Findings & Concerns

        Provide your assessment using highly professional, clinical language. Conclude with a strong, mandatory medical disclaimer emphasizing that your analysis is an AI-generated assessment for educational demonstration and must be formally verified by a physician.
        """

        config_params = {}
        if "pro" in model_name.lower() or "preview" in model_name.lower():
            config_params["thinking_config"] = types.ThinkingConfig(thinking_level="High")

        response = client.models.generate_content(
            model=model_name,
            contents=[video_file, agent_prompt],
            config=types.GenerateContentConfig(**config_params)
        )

        return response.text

    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass