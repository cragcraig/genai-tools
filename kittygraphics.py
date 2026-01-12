""" Implements the kitty graphics protocol for displaying images inline in terminal emulators."""
import sys
import base64
import select
import termios
import tty
import os
from io import BytesIO
from PIL import Image

def display_pil_image(image: Image.Image):
    """
    Displays an image in the terminal using the Kitty graphics protocol.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64_data = base64.standard_b64encode(buffer.getvalue())

    chunk_size = 4096
    offset = 0
    total_len = len(b64_data)

    while offset < total_len:
        chunk = b64_data[offset : offset + chunk_size]
        offset += chunk_size

        is_last = offset >= total_len
        m_val = 0 if is_last else 1

        # First chunk needs specific format/action keys
        if offset <= chunk_size:
            # a=T (transmit and display), f=100 (PNG)
            header = f"a=T,f=100,m={m_val};"
        else:
            # Subsequent chunks only need the continuation key
            header = f"m={m_val};"

        # Construct and write the escape sequence
        sys.stdout.write(f"\x1b_G{header}{chunk.decode('ascii')}\x1b\\")
        sys.stdout.flush()

    sys.stdout.write("\n")

def supports_kitty_graphics():
    """
    Checks if the current terminal supports the Kitty graphics protocol
    by sending a query escape sequence and listening for a valid response.
    """
    # 1. Verification: We must be connected to a terminal (TTY)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False

    # 2. Setup: Save current terminal settings so we can restore them later
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # 3. Enter 'Raw' mode.
        # This prevents the terminal from echoing the response to the screen
        # and allows us to read bytes immediately without waiting for a newline.
        tty.setraw(fd)

        # 4. Send the Query Command.
        # \x1b_G = Start Graphics Command
        # i=42   = An arbitrary ID we define (to recognize the response)
        # a=q    = Action: Query (ask "do you support this?")
        # \x1b\\ = String Terminator (ST)
        query = "\x1b_Gi=42,a=q;\x1b\\"
        sys.stdout.write(query)
        sys.stdout.flush()

        # 5. Listen for response with a timeout.
        # If the terminal doesn't support it, it likely won't respond at all.
        # We wait 0.1 seconds.
        r, _, _ = select.select([sys.stdin], [], [], 0.1)

        if not r:
            # No response received -> Protocol likely not supported
            return False

        # 6. Read the response
        response = ""
        while True:
            # Read one byte at a time
            char = sys.stdin.read(1)
            response += char

            # The response will end with the String Terminator (\x1b\\)
            # Or we stop if it gets suspiciously long to prevent infinite loops
            if response.endswith('\x1b\\') or len(response) > 50:
                break

        # 7. Validate Response
        # A successful response looks like: \x1b_Gi=42;OK\x1b\\
        # We check for our ID (i=42) and the "OK" flag.
        if "i=42" in response and "OK" in response:
            return True

        return False

    except Exception:
        # Fallback for any unexpected I/O errors
        return False

    finally:
        # 8. Cleanup: Always restore original terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    # Create a dummy image if you don't have one
    # For this example, we generate a small red gradient image using PIL
    img = Image.new("RGB", (100, 50), color="red")
    #img = Image.open("out-c9b3/img_0.png")

    print("Displaying PIL image:")
    display_pil_image(img)
