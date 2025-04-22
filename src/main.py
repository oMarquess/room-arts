from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from google import genai
from google.genai import types
# Fix the imports
# import google.generativeai as genai
# from google.generativeai import types
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure Google AI Client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY environment variable not set. Image generation will fail.")
else:
    logger.info("Google API key configured successfully")
    client = genai.Client(api_key=api_key)  # Use this configuration method

async def generate_image_with_gemini(image_bytes: bytes, prompt: str) -> tuple[bytes | None, str | None, str | None]:
    """
    Generates an image using Google Gemini based on an input image and prompt.

    Returns a tuple: (generated_image_bytes, media_type, text_response) or (None, None, None) if generation fails.
    The text_response contains any text returned by the model.
    """
    if not api_key:
        logger.error("Cannot generate image: GOOGLE_API_KEY is not configured.")
        raise HTTPException(status_code=500, detail="Image generation service is not configured.")

    try:
        logger.info(f"Generating image with Gemini. Prompt: '{prompt}'")
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Use the pre-configured client instead of creating a new one
        model = "gemini-2.0-flash-exp-image-generation"
        
        # Generate content using the specified model
        response = client.models.generate_content(
            model = "gemini-2.0-flash-exp-image-generation",
            contents=[prompt, input_image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                # response_mime_types=["image/png"]
            )
        )
        # print(response)


        # response = client.models.generate_content(
        #     model="gemini-2.0-flash-exp-image-generation",
        #     contents=[text_input, image],
        #     config=types.GenerateContentConfig(
        #     response_modalities=['TEXT', 'IMAGE']
        #     )
        # )
        
        # Process the response parts
        text_response = None
        image_data = None
        image_mime_type = None
        
        if hasattr(response, 'candidates') and response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Check for text
                if hasattr(part, 'text') and part.text is not None:
                    text_response = part.text
                    logger.info(f"Received text response: {text_response}")
                
                # Check for image
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    image_data = part.inline_data.data
                    image_mime_type = part.inline_data.mime_type
                    logger.info(f"Received image: {image_mime_type}, size: {len(image_data)} bytes")
        
        if image_data is None:
            logger.warning("No image data found in the response")
            return None, None, text_response
            
        return image_data, image_mime_type, text_response

    except Exception as e:
        logger.exception(f"Error during Gemini image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@app.post("/generate")
async def generate_image_endpoint(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Endpoint to generate an image based on an input image and a text prompt.
    Returns both the generated image and any text response from the model.
    """
    try:
        if not image.content_type.startswith("image/"):
             raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        image_bytes = await image.read()
        logger.info(f"Processing image '{image.filename}' ({image.content_type})")

        # Call the image generation function
        generated_image_bytes, generated_media_type, text_response = await generate_image_with_gemini(image_bytes, prompt)

        # Handle case where image generation failed
        if generated_image_bytes is None:
            if text_response:
                # Return just the text if that's all we got
                return {"success": False, "message": "No image generated", "text_response": text_response}
            else:
                # No image or text was generated
                raise HTTPException(status_code=500, detail="Image generation failed to produce any output")

        # Return the generated image as a streaming response
        # For simple clients that just want the image, this is sufficient
        return StreamingResponse(
            io.BytesIO(generated_image_bytes), 
            media_type=generated_media_type,
            headers={"X-Text-Response": text_response or ""}  
        )

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.exception("An unexpected error occurred during image generation.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    logger.info("Uvicorn server stopped.")
