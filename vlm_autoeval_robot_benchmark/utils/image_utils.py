import io

import numpy as np
from PIL import Image


def numpy_array_to_png_bytes(arr: np.ndarray) -> bytes:
    """
    Convert a NumPy array to PNG file bytes, as if it was saved as a PNG and then read with fp.read()

    Args:
        arr: NumPy array with shape (height, width, 3) and dtype uint8

    Returns:
        PNG file bytes
    """
    # Ensure the array is the right shape and type
    if len(arr.shape) != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected array with shape (height, width, 3), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(arr)

    # Create a BytesIO object to store the image bytes
    buffer = io.BytesIO()

    # Save the image to the BytesIO object as PNG
    img.save(buffer, format="PNG")

    # Get the bytes from the BytesIO object
    png_bytes = buffer.getvalue()

    return png_bytes
