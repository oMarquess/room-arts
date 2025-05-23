# Gemini Image Generation API

A simple REST API built with FastAPI that allows you to generate images based on an existing image and a text prompt. This service uses Google's Gemini 2.0 Flash image generation model to transform your images according to your text instructions.

## Features

- **Image Transformation**: Modify existing images based on text instructions
- **REST API**: Simple HTTP endpoint for easy integration
- **CORS Support**: Configured to accept requests from any origin
- **Input Validation**: Ensures only valid image files are processed
- **Error Handling**: Comprehensive error reporting
- **Logging**: Detailed logs for debugging and monitoring

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Google Generative AI Python SDK
- PIL/Pillow
- Python-dotenv
- Google API Key for Gemini models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gemini-image-api.git
cd gemini-image-api
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory with your Google API key:



To obtain a Google API key:
- Visit [Google AI Studio](https://ai.google.dev/)
- Sign up or log in
- Create an API key under the API section

## Usage

Start the server:

```bash
python src/main.py
```

This will start the server at `http://localhost:8000`.

### API Endpoints

#### POST /generate

Generates a new image based on an input image and text prompt.

**Request Parameters**:
- `prompt` (form field, required): Text instructions describing the desired image transformation
- `image` (file upload, required): The source image to be modified

**Response**:
- 200 OK: Returns the generated image as binary data
- 400 Bad Request: Invalid file type or missing parameters
- 500 Internal Server Error: Server error or model generation error

HTTP headers in the response:
- `X-Text-Response`: Any text response generated by the AI model

## Example Usage

### Using cURL

```bash
curl -X POST \
  http://localhost:8000/generate \
  -F "prompt=Add a blue sofa to this living room" \
  -F "image=@/path/to/your/livingroom.jpg" \
  --output generated_image.png
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/generate"
prompt = "Add a blue sofa to this living room"
image_path = "/path/to/your/livingroom.jpg"

with open(image_path, "rb") as image_file:
    files = {
        "image": (image_path, image_file, "image/jpeg"),
        "prompt": (None, prompt)
    }
    response = requests.post(url, files=files)

if response.status_code == 200:
    # Save the generated image
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    
    # Check for text response
    text_response = response.headers.get("X-Text-Response")
    if text_response:
        print("AI Comment:", text_response)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Using JavaScript/Fetch API

```javascript
async function generateImage() {
    const formData = new FormData();
    formData.append('prompt', 'Add a blue sofa to this living room');
    formData.append('image', document.querySelector('#imageInput').files[0]);
    
    try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            body: formData,
        });
        
        if (response.ok) {
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            document.querySelector('#resultImage').src = imageUrl;
            
            // Check for text response
            const textResponse = response.headers.get('X-Text-Response');
            if (textResponse) {
                document.querySelector('#aiComment').textContent = textResponse;
            }
        } else {
            console.error('Error:', response.status);
            const errorText = await response.text();
            console.error(errorText);
        }
    } catch (error) {
        console.error('Request failed:', error);
    }
}
```

## Deployment

### Docker

A Dockerfile is provided for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]
```

Build and run the Docker container:

```bash
docker build -t gemini-image-api .
docker run -p 8000:8000 --env-file .env gemini-image-api
```

### Heroku

```bash
heroku create gemini-image-api
heroku config:set GOOGLE_API_KEY=your_google_api_key_here
git push heroku main
```

## Model Information

This API uses Google's `gemini-2.0-flash-exp-image-generation` model, which is specialized for:
- Adding or modifying elements in existing images
- Understanding spatial relationships in images
- Generating realistic transformations based on text prompts

## Limitations

- The model may not perfectly follow all instructions
- Certain sensitive content modifications may be rejected
- Very complex scene modifications might not render as expected
- API rate limits may apply based on your Google API key tier

## License

This project is licensed under the MIT License - see the LICENSE file for details.