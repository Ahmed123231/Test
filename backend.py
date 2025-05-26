import os
import time
from datetime import datetime
import re
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from io import BytesIO
import threading
import traceback
from typing import Optional, Tuple, Dict, Any, Callable, List, Tuple
import subprocess
import glob




# Import these modules only if available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) not available - camera functions limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - image processing limited")

try:
    import cloudinary.uploader
    import cloudinary
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    print("Cloudinary not available - upload functions limited")

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False
    print("QRCode not available - QR code generation limited")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Requests not available - API calls limited")

from config import (
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    REPLICATE_API_TOKEN,
    REPLICATE_MODEL_VERSION,
    PHOTO_DIR,
    PROCESSED_DIR,
    QRCODE_DIR,
    ALLOWED_FILENAME_CHARS,
    EID_PROMPT,
    CARTOONIFY_TIMEOUT
)
from thread_utils import ProgressTracker


class PhotoBooth:
    def __init__(self, working_dir=None):
        """Initialize the photo booth with a flexible working directory."""
        self.working_dir = os.getcwd() if working_dir is None else working_dir
        self.user_name = None
        self.current_photo = None
        self.process_start_time = None
        self.camera = None
        # Threading locks
        self.photo_lock = threading.RLock()
        self.api_lock = threading.RLock()
        
        # Progress trackers
        self.cartoonify_progress = ProgressTracker(total_steps=100)
        self.upload_progress = ProgressTracker(total_steps=100)
        self.overall_progress = ProgressTracker(total_steps=100)
        
        # Initialize directories
        try:
            os.makedirs(self.working_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create or access working directory '{self.working_dir}': {str(e)}")

        self.photo_dir = os.path.join(self.working_dir, PHOTO_DIR)
        self.processed_dir = os.path.join(self.working_dir, PROCESSED_DIR)
        self.qrcode_dir = os.path.join(self.working_dir, QRCODE_DIR)

        if CLOUDINARY_AVAILABLE:
            cloudinary.config(
                cloud_name=CLOUDINARY_CLOUD_NAME,
                api_key=CLOUDINARY_API_KEY,
                api_secret=CLOUDINARY_API_SECRET
            )

        os.chdir(self.working_dir)
        print(f"\U0001F4C1 Current working directory set to: {os.getcwd()}")

        for directory in [self.photo_dir, self.processed_dir, self.qrcode_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Failed to create directory '{directory}': {str(e)}")

    def prompt_user_name(self):
        """Prompt the user for their name and sanitize it for filenames."""
        user_name = input("Please enter your name: ").strip()
        if not user_name:
            print("\u26A0 Name cannot be empty. Using 'user' as default.")
            user_name = "user"

        sanitized_name = "".join(c for c in user_name.lower() if c in ALLOWED_FILENAME_CHARS)
        if not sanitized_name:
            print("\u26A0 Invalid name after sanitization. Using 'user' as default.")
            sanitized_name = "user"

        self.user_name = sanitized_name
        print(f"\U0001F464 User name set to: {self.user_name}")
        return self.user_name

    def set_user_name(self, name):
        """Set the user name, sanitizing it for filenames."""
        if not name or not name.strip():
            print("\u26A0 Name cannot be empty. Using 'user' as default.")
            name = "user"

        sanitized_name = "".join(c for c in name.lower() if c in ALLOWED_FILENAME_CHARS)
        if not sanitized_name:
            print("\u26A0 Invalid name after sanitization. Using 'user' as default.")
            sanitized_name = "user"

        self.user_name = sanitized_name
        print(f"\U0001F464 User name set to: {self.user_name}")
        return self.user_name


    def _get_video_capture_devices(self) -> List[Tuple[str, str]]:

        result = []

        try:
            output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
            lines = output.strip().splitlines()
            current_name = ""
            for line in lines:
                if not line.startswith("\t") and line.strip():
                    current_name = line.strip()
                elif line.strip().startswith("/dev/video"):
                    dev_path = line.strip()
                    if "USB" in current_name or "HDMI" in current_name or "Capture" in current_name:
                        result.append((dev_path, current_name))
        except Exception as e:
            print(f"? v4l2-ctl failed: {e}")
            # Fallback: just return /dev/video0..9 that actually exist
            for i in range(4):
                path = f"/dev/video{i}"
                if os.path.exists(path):
                    result.append((path, "Unknown"))

        return result



    def close_camera(self):
        try:
            if self.camera and hasattr(self.camera, 'release'):
                self.camera.release()
                print("? Camera released.")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        self.camera = None



    def is_camera_open(self):
        return self.camera is not None and self.camera.isOpened()







    def initialize_camera(self):
        gst_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "image/jpeg, width=640, height=480, framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("? Failed to open GStreamer pipeline via OpenCV")
            return None

        print("? GStreamer camera opened successfully")
        return cap



    
    def read_frame(self, cap):
    
        if cap and hasattr(cap, 'read'):
            ret, frame = cap.read()
            if ret:
                return frame
        return None

    
    
    
    
    
    




            
    def get_test_frame(self):
        """Generate a test frame when no camera is available."""
        if not NUMPY_AVAILABLE:
            # If numpy isn't available, return a simple PIL image
            test_img = Image.new('RGB', (640, 480), color='white')
            draw = ImageDraw.Draw(test_img)
            draw.text((320, 240), "Test Mode - No Camera", 
                     fill="black", anchor="mm")
            return test_img

        # Create a gradient background
        width, height = 640, 480
        test_img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # Create a green gradient
        for y in range(height):
            # Calculate gradient color (green to light green)
            green = int(100 + (150 * y / height))
            color = (220, green, 220)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add text
        try:
            font = ImageFont.truetype("Arial", 24)
        except:
            font = ImageFont.load_default()
            
        draw.text((width//2-150, height//2-50), "Test Mode - No Camera", 
                 fill="black")
        draw.text((width//2-150, height//2+20), "Photo Booth Ready", 
                 fill="#007F3F")
        
        # Draw a face outline to simulate a person
        face_x, face_y = width//2, height//2-80
        face_radius = 50
        draw.ellipse((face_x-face_radius, face_y-face_radius, 
                     face_x+face_radius, face_y+face_radius), 
                    outline="#007F3F", width=2)
        
        # Add eyes and smile
        draw.ellipse((face_x-25, face_y-20, face_x-10, face_y-5), fill="black")
        draw.ellipse((face_x+10, face_y-20, face_x+25, face_y-5), fill="black")
        draw.arc((face_x-30, face_y, face_x+30, face_y+30), 0, 180, fill="black", width=2)
        
        # Convert to OpenCV format (RGB to BGR) if OpenCV is available
        if CV2_AVAILABLE:
            frame_array = np.array(test_img)
            frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            
            # Add timestamp to make it seem live
            timestamp = f"Test Frame: {datetime.now().strftime('%H:%M:%S')}"
            cv2.putText(frame, timestamp, (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
            
            return frame
        else:
            # Add timestamp using PIL if OpenCV is not available
            timestamp = f"Test Frame: {datetime.now().strftime('%H:%M:%S')}"
            draw.text((10, height-20), timestamp, fill="#009600", font=ImageFont.load_default())
            return test_img

    def capture_photo(self, frame=None):
        """Capture a photo from a provided frame or from the camera."""
        with self.photo_lock:
            if frame is not None:
                # Convert OpenCV frame (BGR) to PIL Image (RGB) if needed
                if CV2_AVAILABLE and isinstance(frame, np.ndarray):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.current_photo = Image.fromarray(frame_rgb)
                elif isinstance(frame, Image.Image):
                    self.current_photo = frame
                else:
                    print("\u26A0 Unsupported frame format for capture")
                    return None, None
            else:
                print("\u26A0 No frame provided for capture")
                return None, None
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user = self.user_name if self.user_name else "user"
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(self.photo_dir, filename)
            
            # Save the photo
            try:
                self.current_photo.save(filepath)
                print(f"📸 Photo captured and saved to {os.path.abspath(filepath)}")
                return self.current_photo, filepath
            except Exception as e:
                print(f"\u274C Error saving photo: {e}")
                traceback.print_exc()
                return None, None

    def get_current_photo(self):
        """Get the current photo (thread-safe)."""
        with self.photo_lock:
            return self.current_photo

    def upload_to_cloudinary(self, filepath, progress_callback=None):
        """Upload an image to Cloudinary and return its URL."""
        if not CLOUDINARY_AVAILABLE:
            if progress_callback:
                progress_callback((0, 100, 0, "Cloudinary not available"))
            print("\u26A0 Cloudinary not available - upload function disabled")
            return None

        with self.api_lock:
            try:
                # Initialize progress
                self.upload_progress.start()
                if progress_callback:
                    self.upload_progress.register_callback(progress_callback)
                
                self.upload_progress.update(message="Preparing to upload...")
                
                # Perform the upload
                self.upload_progress.update(20, message="Uploading to cloud...")
                response = cloudinary.uploader.upload(filepath)
                
                self.upload_progress.update(80, message="Processing upload...")
                url, _ = cloudinary.utils.cloudinary_url(response["public_id"], format=response["format"])
                
                self.upload_progress.complete(f"Uploaded to: {url}")
                print(f"\u2601\ufe0f Uploaded to Cloudinary: {url}")
                return url
            except Exception as e:
                self.upload_progress.update(message=f"Error: {str(e)}")
                print(f"\u274C Error uploading to Cloudinary: {str(e)}")
                traceback.print_exc()
                return None

    def cartoonify_image(self, image_path, progress_callback=None):
        """Cartoonify an image with Eid al-Adha themes using Replicate."""
        if not REQUESTS_AVAILABLE or not CLOUDINARY_AVAILABLE:
            if progress_callback:
                progress_callback((0, 100, 0, "Required libraries not available"))
            print("\u26A0 Required libraries not available - cartoonify function disabled")
            return None

        with self.api_lock:
            # Initialize progress tracker
            self.cartoonify_progress.start()
            if progress_callback:
                self.cartoonify_progress.register_callback(progress_callback)
            
            self.cartoonify_progress.update(message="Uploading image for processing...")
            
            # First upload image to Cloudinary to get a URL
            image_url = self.upload_to_cloudinary(image_path)
            if not image_url:
                self.cartoonify_progress.update(message="Failed to upload image")
                return None
            
            self.cartoonify_progress.update(10, message="Preparing cartoonify request...")

            headers = {
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }
            json_data = {
                "version": REPLICATE_MODEL_VERSION,
                "input": {
                    "image": image_url,
                    "prompt": EID_PROMPT,
                }
            }
            
            response = None
            try:
                self.cartoonify_progress.update(20, message="Starting cartoonification...")
                response = requests.post(
                    "https://api.replicate.com/v1/predictions", 
                    json=json_data, 
                    headers=headers
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                error_msg = f"Failed to start prediction: {str(e)}"
                self.cartoonify_progress.update(message=error_msg)
                print(f"\u274C {error_msg}")
                print(f"Response: {response.json() if response else 'No response'}")
                return None
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error during prediction: {str(e)}"
                self.cartoonify_progress.update(message=error_msg)
                print(f"\u274C {error_msg}")
                return None

            if not response:
                self.cartoonify_progress.update(message="Failed to get response")
                return None

            prediction = response.json()
            urls = prediction.get("urls")
            prediction_url = urls.get("get") if urls else None
            
            if not prediction_url:
                self.cartoonify_progress.update(message="Invalid API response")
                return None

            self.cartoonify_progress.update(25, message="Processing image...")
            print("\u23F3 Waiting for model to finish processing...")
            
            # Polling with progress updates
            start_time = time.time()
            attempt = 0
            max_attempts = 60  # Set a reasonable limit
            
            while attempt < max_attempts:
                attempt += 1
                
                if attempt > 1:
                    # Wait between polling attempts, with increasing delays
                    wait_time = min(3, 0.5 * attempt)  # Cap at 3 seconds
                    time.sleep(wait_time)
                
                try:
                    result = requests.get(prediction_url, headers=headers).json()
                except Exception as e:
                    print(f"Error polling prediction status: {e}")
                    continue
                
                status = result.get("status")
                
                # Calculate progress based on status
                progress = 25  # Starting progress
                message = "Processing image..."
                
                if status == "succeeded":
                    output_url = result.get("output")
                    if output_url:
                        # Download the cartoonified image
                        self.cartoonify_progress.update(90, message="Downloading result...")
                        
                        try:
                            response = requests.get(output_url)
                            response.raise_for_status()
                            
                            # Save the cartoonified image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            user = self.user_name if self.user_name else "user"
                            filename = f"{timestamp}.jpg"
                            filepath = os.path.join(self.processed_dir, filename)
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            # Also load as PIL Image
                            cartoon_image = Image.open(BytesIO(response.content))
                            
                            self.cartoonify_progress.complete(f"Cartoonification complete: {filepath}")
                            print(f"🎨 Cartoonification complete: {os.path.abspath(filepath)}")
                            
                            return cartoon_image, filepath, output_url
                        except Exception as e:
                            self.cartoonify_progress.update(message=f"Error downloading result: {str(e)}")
                            print(f"\u274C Error downloading cartoonified image: {str(e)}")
                            traceback.print_exc()
                            return None
                    else:
                        self.cartoonify_progress.update(message="Cartoonification succeeded but no output URL")
                        return None
                elif status == "failed":
                    error = result.get("error")
                    self.cartoonify_progress.update(message=f"Cartoonification failed: {error}")
                    print(f"\u274C Cartoonification failed: {error}")
                    return None
                elif status == "processing":
                    # Update progress based on elapsed time
                    elapsed = time.time() - start_time
                    progress_percent = min(85, 25 + (elapsed / CARTOONIFY_TIMEOUT) * 60)
                    self.cartoonify_progress.update(int(progress_percent), message="Processing image...")
                
                # Check for timeout
                if time.time() - start_time > CARTOONIFY_TIMEOUT:
                    self.cartoonify_progress.update(message="Operation timed out")
                    print("\u274C Cartoonification timed out")
                    return None
            
            self.cartoonify_progress.update(message="Maximum attempts reached")
            print("\u274C Maximum polling attempts reached")
            return None

    def generate_qr_code(self, url):
        """Generate a QR code for the given URL and save it."""
        if not QRCODE_AVAILABLE:
            print("\u26A0 QRCode not available - QR code generation disabled")
            return None

        try:
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)

            # Create an image from the QR Code
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Save the QR code
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user = self.user_name if self.user_name else "user"
            f"{timestamp}.jpg"
            filename = "qr_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
            filepath = os.path.join(self.qrcode_dir, filename)
            
            qr_img.save(filepath)
            print(f"🔳 QR Code generated and saved to {os.path.abspath(filepath)}")
            
            # Return only the filepath to avoid tuple handling issues
            return filepath
        except Exception as e:
            print(f"\u274C Error generating QR code: {str(e)}")
            traceback.print_exc()
            return None

    def process_photo(self, frame, progress_callback=None):
        """Process a photo with cartoonification and QR code generation."""
        self.overall_progress.start()
        if progress_callback:
            self.overall_progress.register_callback(progress_callback)
        
        self.process_start_time = time.time()
        self.overall_progress.update(0, message="Starting photo processing...")
        
        # Custom progress callbacks for individual operations
        def cartoonify_progress_update(progress_data):
            step, total, percentage, message = progress_data
            # Map to 0-70% of overall progress
            overall_percent = int((percentage / 100) * 70)
            self.overall_progress.update(overall_percent, message=f"Cartoonify: {message}")
        
        def upload_progress_update(progress_data):
            step, total, percentage, message = progress_data
            # Map to 70-90% of overall progress
            overall_percent = 70 + int((percentage / 100) * 20)
            self.overall_progress.update(overall_percent, message=f"Upload: {message}")
        
        # Step 1: Capture the photo
        self.overall_progress.update(5, message="Capturing photo...")
        photo_result = self.capture_photo(frame)
        if not photo_result or photo_result[0] is None:
            self.overall_progress.update(message="Failed to capture photo")
            return None
        
        photo, photo_path = photo_result
        
        # Step 2: Cartoonify the image
        self.overall_progress.update(10, message="Starting cartoonification...")
        cartoon_result = self.cartoonify_image(photo_path, cartoonify_progress_update)
        
        # For testing or if cartoonification fails, use the original photo
        if not cartoon_result:
            self.overall_progress.update(70, message="Using original photo (cartoonify failed)")
            cartoon_url = self.upload_to_cloudinary(photo_path, upload_progress_update)
            if not cartoon_url:
                self.overall_progress.update(message="Failed to upload photo")
                return None
            
            result_image = photo
            result_path = photo_path
        else:
            cartoon_image, cartoon_path, cartoon_url = cartoon_result
            result_image = cartoon_image
            result_path = cartoon_path
        
        # Step 3: Generate QR code
        self.overall_progress.update(90, message="Generating QR code...")
        qr_result = self.generate_qr_code(cartoon_url) if cartoon_url else None
        
        # Step 4: Finalize
        self.overall_progress.update(95, message="Finalizing...")
        
        # Put everything together in a results dictionary
        results = {
            "original_photo": (photo, photo_path),
            "processed_image": (result_image, result_path),
            "image_url": cartoon_url,
            "qr_code": qr_result,
            "processing_time": time.time() - self.process_start_time
        }
        
        self.overall_progress.complete("Processing complete")
        return results

    def get_processing_progress(self):
        """Get current progress of any ongoing processing."""
        return self.overall_progress.get_progress()