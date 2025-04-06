import base64
from mimetypes import guess_type
def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string with MIME type."""
    try:
        mime_type, _ = guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            print(f"Warning: Could not determine image MIME type for {image_path}. Defaulting to 'image/jpeg'.")
            mime_type = 'image/jpeg'

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None