"""Image wrapper classes for Nano Banana.

Provides a common interface for working with images from different sources
(PIL Images and Google GenAI API responses).
"""

import io
from google.genai import types
from PIL import Image


def pil_image_to_png_bytes(img):
    """Helper to convert PIL Image to bytes for the API."""
    with io.BytesIO() as output:
        img.save(output, 'PNG')
        return output.getvalue()


class PILImageWrapper:
    """Wrapper around a (de-facto standard) PIL Image.

    Exposes a common wrapper Image interface that can also be implemented for
    the types.Image which is returned from Nano Banana image generation requests
    via the google.genai library.
    """

    def __init__(self, img):
        assert (img is not None)
        self.img = img

    def as_google_genai_types_part(self):
        return types.Part(
            inline_data=types.Blob(
                mime_type='image/png',
                data=pil_image_to_png_bytes(self.img)
            )
        )

    def as_pil_image(self):
        return self.img

    def save(self, path_or_bytes, format=None):
        return self.img.save(path_or_bytes, format)

    def show(self):
        return self.img.show()


class GoogleGenAITypesImageWrapper:
    """Wrapper for a google.genai.types.Part representing an image part.

    The types.Part is easily converted to a types.Image by
    types.Part.as_image()

    """

    def __init__(self, part):
        self.part = part
        self.img = part.as_image()
        assert (self.img is not None)

    def as_google_genai_types_part(self):
        return self.part

    def as_pil_image(self):
        return Image.open(io.BytesIO(self.img.image_bytes))

    def save(self, path):
        return self.img.save(path)

    def show(self):
        return self.img.show()
