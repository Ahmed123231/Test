import tkinter as tk
from tkinter import ttk, PhotoImage, font
import math
import random
import threading
import time
from datetime import datetime
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageColor, ImageFont
from typing import Optional, Dict, Any, List, Tuple, Callable
import queue
import traceback

try:
    import ttkthemes as tt
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

from backend import PhotoBooth
from thread_utils import ThreadPool, CameraThread, ProgressTracker


class DPIScaler:
    """Utility class for DPI scaling calculations."""
    #def __init__(self):
        # Cache for scale factors to avoid recalculating
        #self.scale_cache = {}
        
    @staticmethod
    def get_dpi_scale(widget):
        """Get the DPI scaling factor."""
        try:
            root = widget.winfo_toplevel()
            # Use a simpler approach that works cross-platform
            scale = root.winfo_fpixels('1i') / 96.0
            # Ensure scale is reasonable (not too small or too large)
            return max(1.0, min(2, scale))
        except:
            # Default scale if all else fails
            return 1.0
    
    def scale(self, widget, value):
        """Scale a value according to DPI."""
        # Use cached scale value if available
        #widget_id = str(widget)
        #if widget_id not in self.scale_cache:
            #self.scale_cache[widget_id] = self.get_dpi_scale(widget)
        
        # Calculate and ensure it's at least 1
        return max(1, int(value * self.get_dpi_scale(widget))) 
    
    def font_size(self, widget, size):
        """Get a scaled font size."""
        # Use cached scale value if available
        #widget_id = str(widget)
        #if widget_id not in self.scale_cache:
            
        # Calculate and ensure minimum reasonable font size
        return max(1, int(size * self.get_dpi_scale(widget))) 


class CustomButton(tk.Canvas):
    def __init__(self, parent, text, command, width=180, height=45,
                 bg_normal="#FFFFFF", bg_hover="#EAFFD5", bg_active="#C4F8C5",
                 text_color="#007F3F", **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg="#F0F0F0", **kwargs)

        self.command = command
        self.text = text
        self.width = width
        self.height = height

        self.normal_img = self._create_button_image(bg_normal)
        self.hover_img = self._create_button_image(bg_hover)
        self.active_img = self._create_button_image(bg_active)

        self.image_id = self.create_image(width//2, height//2, image=self.normal_img)
        self.text_id = self.create_text(width//2, height//2, text=text, font=("Arial", 14, "bold"), fill=text_color)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _create_button_image(self, fill_color):
        scale = 3
        img = Image.new("RGBA", (self.width * scale, self.height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        radius = int(self.width * 0.08 * scale)
        border_width = 2 * scale

        draw.rounded_rectangle(
            [(border_width, border_width),
             (self.width * scale - border_width, self.height * scale - border_width)],
            radius=radius,
            fill=fill_color,
            outline="#007F3F",
            width=border_width
        )

        img = img.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(master=self, image=img)

    def _on_enter(self, event):
        self.itemconfig(self.image_id, image=self.hover_img)

    def _on_leave(self, event):
        self.itemconfig(self.image_id, image=self.normal_img)

    def _on_click(self, event):
        self.itemconfig(self.image_id, image=self.active_img)

    def _on_release(self, event):
        self.itemconfig(self.image_id, image=self.hover_img)
        if self.command:
            self.command()

    def set_state(self, state):
        if state == "disabled":
            self.unbind("<Enter>")
            self.unbind("<Leave>")
            self.unbind("<Button-1>")
            self.unbind("<ButtonRelease-1>")
            self.itemconfig(self.image_id, image=self.normal_img)
            self.itemconfig(self.text_id, fill="#999999")
        else:
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<Button-1>", self._on_click)
            self.bind("<ButtonRelease-1>", self._on_release)
            self.itemconfig(self.image_id, image=self.normal_img)
            self.itemconfig(self.text_id, fill="#007F3F")


class CustomEntry(tk.Frame):
    """A custom entry field with placeholder, error states, and DPI awareness."""
    def __init__(self, parent, placeholder="", width=300, **kwargs):
        tk.Frame.__init__(self, parent, bg="#F0F0F0", **kwargs)
        
        # Initialize DPI scaler
        self._dpi_scaler = DPIScaler()
        
        # Scale dimensions
        self.width = self._dpi_scaler.scale(parent, width)
        self.height = self._dpi_scaler.scale(parent, 40)
        
        # Store references
        self._image_references = []
        self.placeholder = placeholder
        self.placeholder_color = "gray"
        self.default_fg_color = "black"
        
        # Create canvas
        self.bg_canvas = tk.Canvas(self, width=self.width, height=self.height,
                                  highlightthickness=0, bg="#F0F0F0")
        self.bg_canvas.pack(fill="both", expand=True)
        
        # Create entry backgrounds
        self.normal_bg = self._create_entry_bg("white", "#EAFFD5")
        self.focus_bg = self._create_entry_bg("white", "#007F3F")
        self.error_bg = self._create_entry_bg("white", "red")
        
        if not self.normal_bg:
            raise ValueError("Failed to create normal_bg PhotoImage")
        
        self.bg_img = self.normal_bg
        self.bg = self.bg_canvas.create_image(self.width/2, self.height/2, 
                                             image=self.bg_img)
        
        # Calculate safe area inside rounded rectangle
        border_width = max(2, int(self.width * 0.007))
        radius = int(self.height * 0.4)
        # Reduce width to stay within rounded corners + border
        safe_width = self.width - (border_width * 2) - (radius)  
        
        # Create entry widget
        font_size = self._dpi_scaler.font_size(parent, 14)
        
        # Create frame with transparent background for entry
        entry_frame = tk.Frame(self, bg="#007F3F", highlightthickness=0, bd=0)
        
        # Calculate character width based on pixel width
        # Use a more conservative estimation to fit within the rounded corners
        approx_char_width = font_size * 0.7  # Average character width in pixels
        char_width = max(1, int(safe_width / approx_char_width))
        
        self.entry = tk.Entry(
            entry_frame, 
            font=("Tajawal", font_size), 
            bd=0,
            width=char_width,
            highlightthickness=0,
            bg="white"  # Match the background image fill
        )
        self.entry.pack(fill="none", expand=False)
        self.entry.configure(justify='center')
        
        # Place entry in center of rounded background
        self.entry_window = self.bg_canvas.create_window(
            self.width//2, self.height//2,
            window=entry_frame, 
            width=safe_width
        )
        
        self._put_placeholder()
        
        self.entry.bind("<FocusIn>", self._focus_in)
        self.entry.bind("<FocusOut>", self._focus_out)

    def _create_entry_bg(self, fill_color, outline_color):
        """Create a background image for the entry field."""
        try:
            # Create image using PIL directly
            img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            radius = int(self.height * 0.4)
            border_width = max(2, int(self.width * 0.007))
            
            draw.rounded_rectangle(
                [(border_width, border_width), 
                 (self.width-border_width, self.height-border_width)],
                radius=radius,
                fill=fill_color,
                outline=outline_color,
                width=border_width
            )
            
            if fill_color == "white":
                shadow_color = "#00000015"
                shadow_offset = max(1, int(border_width * 0.5))
                draw.rounded_rectangle(
                [(border_width+shadow_offset, border_width+shadow_offset),
                (self.width-border_width-shadow_offset, self.height-border_width-shadow_offset)],
                radius=radius-shadow_offset,
                outline=shadow_color,
                width=shadow_offset
            )
            
            photo = ImageTk.PhotoImage(master=self.bg_canvas, image=img)
            self._image_references.append(photo)
            return photo
        except Exception as e:
            print(f"Error in create_entry_bg: {e}")
            return None

    def _put_placeholder(self):
        self.entry.delete(0, "end")
        self.entry.insert(0, self.placeholder)
        self.entry.config(fg=self.placeholder_color)

    def _focus_in(self, event):
        if self.entry.get() == self.placeholder:
            self.entry.delete(0, "end")
            self.entry.config(fg=self.default_fg_color)
        self.bg_canvas.itemconfig(self.bg, image=self.focus_bg)

    def _focus_out(self, event):
        if not self.entry.get():
            self._put_placeholder()
        self.bg_canvas.itemconfig(self.bg, image=self.normal_bg)

    def get(self):
        return "" if self.entry.get() == self.placeholder else self.entry.get()

    def show_error(self):
        self.bg_canvas.itemconfig(self.bg, image=self.error_bg)
        self.after(200, lambda: self.bg_canvas.itemconfig(self.bg, image=self.normal_bg))


class DecorationElement(DPIScaler):
    """Base class for decorative elements with DPI awareness."""
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.ids = []
        
        # Store the scaling factor for consistent scaling across all methods
        self.dpi_factor = self.get_dpi_scale(canvas)
    
    def scale_value(self, value):
        """Scale a value based on the stored DPI factor."""
        return int(value * self.dpi_factor)
    
    def create_image_buffer(self, width, height):
        """Create a properly sized image buffer with DPI awareness."""
        # Create a larger buffer for high DPI displays (2x or 3x depending on scale)
        multiplier = max(2, round(self.dpi_factor * 2))
        scaled_width = width * multiplier
        scaled_height = height * multiplier
        
        # Create a new transparent image
        return Image.new('RGBA', (scaled_width, scaled_height), (0, 0, 0, 0))
    
    def remove(self):
        """Remove all canvas elements."""
        for item_id in self.ids:
            try:
                self.canvas.delete(item_id)
            except:
                pass  # Handle possible errors if canvas no longer exists
        self.ids = []



class ProgressBar(tk.Canvas):
    """Custom progress bar with animation and text display."""
    def __init__(self, parent, width=400, height=30, **kwargs):
        self._dpi_scaler = DPIScaler()

        # Scale dimensions
        self.width = self._dpi_scaler.scale(parent, width)
        self.height = self._dpi_scaler.scale(parent, height)
        self.border_width = max(1, int(self.height * 0.1))

        # ✅ First initialize the Canvas
        super().__init__(
            parent,
            width=self.width,
            height=self.height,
            highlightthickness=0,
            **kwargs
        )

        # ✅ Now safe to bind events
        self.bind("<Configure>", self._on_resize)
        
        self.progress = 0
        self.text = ""
        self.animation_speed = 3
        self.color_bg = "#E0E0E0"
        self.color_fg = "#007F3F"
        self.color_border = "#007F3F"
        self.is_active = False
        self.after_id = None
        self.target_progress = 0
        self.current_progress_width = 0
        
        # Draw initial components
        radius = self.height // 3
        self.bg_id = self.create_rounded_rect(
            self.border_width, 
            self.border_width, 
            self.width - self.border_width, 
            self.height - self.border_width,
            radius=radius,
            fill=self.color_bg, 
            outline=self.color_border,
            width=self.border_width
        )
        
        self.progress_id = self.create_rounded_rect(
            self.border_width * 2, 
            self.border_width * 2, 
            self.border_width * 2, 
            self.height - self.border_width * 2,
            radius=radius - self.border_width,
            fill=self.color_fg, 
            outline=""
        )
        
        # Text display
        font_size = self._dpi_scaler.font_size(parent, 12)
        self.text_id = self.create_text(
            self.width // 2, 
            self.height // 2,
            text="",
            font=("Tajawal", font_size),
            fill="#000000"
        )
        
        # Set up animation
        self.target_progress = 0
        self.current_progress_width = 0
    def _draw_static_elements(self):
        radius = self.height // 3
        self.bg_id = self.create_rounded_rect(
            self.border_width,
            self.border_width,
            self.width - self.border_width,
            self.height - self.border_width,
            radius=radius,
            fill=self.color_bg,
            outline=self.color_border,
            width=self.border_width
    )

        self.progress_id = self.create_rounded_rect(
            self.border_width * 2,
            self.border_width * 2,
            self.border_width * 2,
            self.height - self.border_width * 2,
            radius=radius - self.border_width,
            fill=self.color_fg,
            outline=""
    )

        font_size = self._dpi_scaler.font_size(self, 12)
        self.text_id = self.create_text(
            self.width // 2,
            self.height // 2,
            text=self.text,
            font=("Tajawal", font_size),
            fill="#000000"
    )
    def _on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.border_width = max(1, int(self.height * 0.1))
        self.delete("all")  # Clear current drawings
        self._draw_static_elements()
        self._animate_progress()

    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        """Create a rounded rectangle on the canvas."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set_progress(self, progress, text=None):
        """Set the progress value (0-100) and optionally update text."""
        self.target_progress = max(0, min(100, progress))
        
        if text is not None:
            self.text = text
            self.itemconfig(self.text_id, text=text)
        
        # Start animation if not already running
        if not self.is_active:
            self.is_active = True
            self._animate_progress()
    
    def _animate_progress(self):
        """Animate the progress bar smoothly."""
        progress_width = (self.width - (self.border_width * 4)) * (self.target_progress / 100)
        x2 = self.border_width * 2 + max(0, self.current_progress_width)
        self.coords(self.text_id, self.width // 2, self.height // 2)

        if not self.is_active:
            return
        
        # Calculate target width based on progress percentage
        progress_width = (self.width - (self.border_width * 4)) * (self.target_progress / 100)
        
        # Animate smoothly
        if abs(self.current_progress_width - progress_width) < 1:
            self.current_progress_width = progress_width
            self.is_active = False
        else:
            # Move towards target
            diff = (progress_width - self.current_progress_width) / self.animation_speed
            self.current_progress_width += diff
        
        # Update progress bar
        x2 = self.border_width * 2 + max(0, self.current_progress_width)
        self.coords(
            self.progress_id,
            self.border_width * 2, 
            self.border_width * 2, 
            x2, 
            self.height - self.border_width * 2
        )
        
        # Continue animation if needed
        if self.is_active:
            self.after_id = self.after(16, self._animate_progress)  # ~60fps
    
    def reset(self):
        """Reset the progress bar to 0%."""
        self.set_progress(0, "")
    
    def stop_animation(self):
        """Stop any running animation."""
        self.is_active = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None


class PhotoPreviewCanvas(tk.Canvas):
    def __init__(self, parent, width=640, height=480, **kwargs):
        self.width = width
        self.height = height
        self.latest_frame = None
        super().__init__(parent, width=width, height=height, **kwargs)
        self.bind("<Configure>", self._on_resize)

        self.photo_image = None
        self.placeholder_id = self.create_text(
            self.width // 2,
            self.height // 2,
            text="Camera initializing...",
            font=("Arial", 16),
            fill="#333333"
        )
        self.image_id = None
        self.photo_image = None
        
        self.draw_decorative_border()
    
    def _on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.config(width=self.width, height=self.height)
    
        self.delete("border")
        self.draw_decorative_border()
    
        # ✅ Re-render the image to match new size
        if hasattr(self, 'latest_frame') and self.latest_frame is not None:
            self.update_preview(self.latest_frame)

    def draw_decorative_border(self):
        self.create_rectangle(0, 0, self.width, self.height, outline="#007F3F", width=2, tags="border")


    def update_preview(self, frame):
    # Convert OpenCV BGR frame to PIL Image
        self.latest_frame = frame
        try:
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif isinstance(frame, Image.Image):
                image = frame
            else:
                raise ValueError("Unsupported frame format for preview")

            resized = self.resize_to_fit(image)
            self.photo_image = ImageTk.PhotoImage(image=resized)

            if self.image_id:
                self.itemconfig(self.image_id, image=self.photo_image)
                self.coords(self.image_id, 0, 0)

            else:
                self.image_id = self.create_image(
                    0, 0,
                    anchor=tk.NW,  # ✅ top-left anchor instead of CENTER
                    image=self.photo_image
                )

        except Exception as e:
            print(f"Error updating preview: {e}")
            traceback.print_exc()


    def resize_to_fit(self, image):
        canvas_width = max(1, self.winfo_width())
        canvas_height = max(1, self.winfo_height())

        # Image aspect ratio
        image_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height

        # Scale image to fully cover canvas (might crop)
        if image_ratio > canvas_ratio:
            # Fit to height (crop width)
            new_height = canvas_height
            new_width = int(canvas_height * image_ratio)
        else:
            # Fit to width (crop height)
            new_width = canvas_width
            new_height = int(canvas_width / image_ratio)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop center to match canvas size
        left = (new_width - canvas_width) // 2
        top = (new_height - canvas_height) // 2
        right = left + canvas_width
        bottom = top + canvas_height

        return resized.crop((left, top, right, bottom))



class PhotoBoothGUI(tk.Tk):
    """Main Photo Booth GUI application."""
    def __init__(self, booth_backend):
        self.countdown_label = None
        self.countdown_seconds = 10
        self.countdown_after_id = None
        self.is_counting_down = False
        self.is_preview_frozen = False
        self.is_processing = False
        super().__init__()
        self.decorations = []
        self.decorations = []
        self.preview_queue = queue.Queue(maxsize=3)
        # Store reference to backend
        self.booth = booth_backend
        
        # Set window properties
        self.title("Eid Photo Booth")
        self.configure(bg="#FFFFFF")
        self.minsize(800, 600)
        # Ensure window decorations (title bar, minimize/maximize/close buttons) are visible
        self.overrideredirect(False)
        
        # Create worker thread pool for background operations
        self.thread_pool = ThreadPool(num_workers=2, name_prefix="Worker")
        
        # Set up camera thread variables
        self.camera_thread = None
        # In __init__ of PhotoBoothGUI class:
        self.preview_queue = queue.Queue(maxsize=2)  # Increased buffer size # Only keep latest frame
        
        # Track application state
        self.current_frame = None
        self.is_capturing = False
        self.is_processing = False
        self.latest_result = None
        self._is_closing = False  # Flag for clean shutdown of threads
        self.simulation_active = False  # Flag for test mode
        
        # Set up UI
        self._create_ui()
        
        # Start the camera preview
        self._start_camera_preview()
        
        # Set up cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_ui(self):
        """Create the user interface."""
        # Use DPI scaler for consistent sizing
        self.dpi_scaler = DPIScaler()
        
        # Main container frame
        main_frame = tk.Frame(self, bg="#d5ebee")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create preview section with minimum size but ability to expand
        preview_section = tk.Frame(main_frame, bg="#F0F0F0")
        preview_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        preview_section.grid_propagate(False)  # Prevent automatic resize based on children
        
        # Configure grid to have a single cell that expands
        preview_section.grid_columnconfigure(0, weight=1)
        preview_section.grid_rowconfigure(0, weight=1)
        
        # Preview frame using grid instead of pack for better control
        preview_frame = tk.Frame(
            preview_section,
            bg="#DDDDDD",
            bd=2,
            relief=tk.GROOVE
        )
        preview_frame.grid(row=0, column=0, sticky="nsew")  # Fill the cell completely
        
        # Canvas for displaying the preview - fixed initial size
        self.preview_canvas = PhotoPreviewCanvas(
            preview_frame,
            width=640,
            height=400,
            bg="black"
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=False)  # Don't allow the canvas to drive expansion
        
        # Store original dimensions for aspect ratio calculations
        self._original_width = 640
        self._original_height = 400
        self._aspect_ratio = self._original_width / self._original_height
        
        # Bottom row containing QR code, capture button, etc.
        bottom_row = tk.Frame(main_frame, bg="#F0F0F0", height=170)
        bottom_row.pack(fill=tk.X, expand=False, pady=(0, 10))
        bottom_row.pack_propagate(False)  # Fix the height of the bottom row
        
        qr_border = tk.Frame(bottom_row, bg="#DDDDDD", bd=2, relief=tk.GROOVE)
        qr_border.pack(side=tk.LEFT, padx=(10, 5))
        
        self.qr_canvas = tk.Canvas(
            qr_border,
            width=150,
            height=150,
            bg="white",
            highlightthickness=0
        )
        self.qr_canvas.pack(padx=5, pady=5)
        
        self.qr_placeholder_id = self.qr_canvas.create_text(
            75, 75,
            text="QR\nwill appear\nhere",
            font=("Arial", 10),
            fill="#777777",
            justify=tk.CENTER
        )
        
        self.capture_btn = ImageButton(
            bottom_row,
            image_path="assets/camera_button.jpg",
            command=self._on_capture_click,
            size=(100, 100)
        )
        self.capture_btn.pack(side=tk.LEFT, padx=(10, 10))
        
        self.countdown_btn = tk.Canvas(
            bottom_row,
            width=100,
            height=100,
            bg="#F0F0F0",
            highlightthickness=0  
        )
        self.countdown_btn.pack(side=tk.LEFT)
        
        scale_factor = 4
        circle_size = 100
        hi_res_size = circle_size * scale_factor
        
        circle_img_hr = Image.new("RGBA", (hi_res_size, hi_res_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(circle_img_hr)
        
        draw.ellipse(
            (12, 12, hi_res_size - 12, hi_res_size - 12),
            fill="#999999",  # gray fill
            outline="#666666",  # darker gray border
            width=12
        )
        
        circle_img = circle_img_hr.resize((circle_size, circle_size), Image.Resampling.LANCZOS)
        self.circle_photo = ImageTk.PhotoImage(circle_img)
        
        self.countdown_btn.create_image(0, 0, anchor=tk.NW, image=self.circle_photo)
        self.countdown_text_id = self.countdown_btn.create_text(
            50, 50,
            text="10",
            font=("Arial", 28, "bold"),
            fill="white"
        )
        self.countdown_btn.itemconfig(self.countdown_text_id, state="hidden")
        
        self.status_label = tk.Label(
            bottom_row,
            text="Ready",
            font=("Arial", self.dpi_scaler.font_size(self, 12)),
            bg="#F0F0F0",
            fg="#555555",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=250
        )
        self.status_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Set up resize handling
        self._resize_job = None
        self.bind("<Configure>", self._on_window_resize)
        # Initial resize after a short delay
        self.after(100, self._on_window_resize)
    
    def _on_window_resize(self, event=None):
        """Handle window resize events to update the preview canvas proportionally"""
        # Only respond to main window resize events, not child widget events
        if event is not None and event.widget != self:
            return
            
        # Cancel previous resize job if it exists
        if hasattr(self, '_resize_job') and self._resize_job:
            self.after_cancel(self._resize_job)
            
        # Schedule a new resize job with a delay to avoid rapid updates
        self._resize_job = self.after(100, self._update_preview_size)
    
    def _update_preview_size(self):
        """Update preview size with controlled dimensions"""
        # Calculate available space
        width = self.winfo_width() - 40  # Account for padding and borders
        height = self.winfo_height() - 220  # Account for bottom controls
        
        # Enforce minimum dimensions
        width = max(width, 320)
        height = max(height, 200)
        
        # Maintain aspect ratio
        target_ratio = self._aspect_ratio
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # Too wide, adjust width based on height
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Too tall, adjust height based on width
            new_width = width
            new_height = int(width / target_ratio)
        
        # Update canvas size directly
        self.preview_canvas.config(width=new_width, height=new_height)
        
        # Force geometry update to prevent recursive resize issues
        self.update_idletasks()


    def _handle_camera_init_result(self, _):
        self.camera_thread = CameraThread(
            camera_init_func=self.booth.initialize_camera,
            frame_callback=self._on_frame_received
        )
        self.camera_thread.start()
        self._log_status("?? Camera preview started")

    #def _handle_camera_init_result(self, camera):


        #if camera is None:
        #  self._log_status("?? No camera detected - using test mode")
        #    test_frame = self.booth.get_test_frame()
        #    if test_frame is not None:
        #        self._on_frame_received(test_frame)
        #    self._start_simulation_mode()
        #    self._set_buttons_state("preview", "normal")
        #else:
        #    self.camera_thread = CameraThread(
        #        camera_init_func=self.booth.initialize_camera,
        #        frame_callback=self._on_frame_received
        #    )
        #    self.camera_thread.start()
        #    self._log_status("?? Camera preview started")        






    def _start_camera_preview(self):
        """Start the camera preview thread."""


        self._handle_camera_init_result(None)
        self._log_status("Initializing camera...")
        # Initialize camera on a worker thread
        #def init_camera_task():
        #    return self.booth.initialize_camera()
        #def init_camera_callback(camera):
        #    self.after(0, self._handle_camera_init_result, camera)
#
        #self.thread_pool.submit(
        #command_type="init_camera",
        #data=init_camera_task,
        #callback=init_camera_callback
        #)
        #self._log_status("Initializing camera...")
            #if camera is None:
             #   self._log_status("⚠️ No camera detected - using test mode")
              #  # Create a test image right away for immediate feedback
               # test_frame = self.booth.get_test_frame()
                #if test_frame is not None:
                 #   self._on_frame_received(test_frame)
                    
                # Start a simulation thread using the test frame generator
                #self._start_simulation_mode()
                # Enable capture button so we can test the workflow
                #self._set_buttons_state("preview", "normal")
                #return
            
            # Now that camera is ready, start the camera thread
            #self.camera_thread = CameraThread(
             #   camera_init_func=self.booth.initialize_camera,
              #  frame_callback=self._on_frame_received
            #)
            #self.camera_thread.start()
            #self._log_status("📸 Camera preview started")
        
        #self.thread_pool.submit(
         #   command_type="init_camera",
          #  data=self.booth.initialize_camera,
           # callback=init_camera_callback
        #)
      
    def _start_simulation_mode(self):
        """Start simulation mode with test frames when no camera is available."""
        # Create test frame simulation thread
        self.simulation_active = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            name="SimulationThread",
            daemon=True
        )
        self.simulation_thread.start()
        self._log_status("📸 Test mode activated - simulating camera")
        
    def _simulation_loop(self):
        """Generate test frames in a loop to simulate camera feed."""
        try:
            # Continue until stopped or window closed
            while self.simulation_active and not self._is_closing:
                try:
                    # Get a test frame
                    test_frame = self.booth.get_test_frame()
                    
                    # Process it like a regular camera frame
                    if test_frame is not None:
                        self._on_frame_received(test_frame)
                        
                        # Log status if first frame
                        if not hasattr(self, '_first_frame_received'):
                            self._first_frame_received = True
                            self._log_status("✅ Test mode active - camera feed simulated")
                            # Enable buttons since preview is working
                            self._set_buttons_state("preview", "normal")
                    
                    # Throttle to simulate realistic camera frame rate
                    time.sleep(1/15)  # ~15 FPS
                except Exception as inner_e:
                    print(f"Error processing test frame: {inner_e}")
                    traceback.print_exc()
                    time.sleep(1)  # Wait a bit before retrying
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            traceback.print_exc()
        
    def _create_test_frame(self):
        """Create a test frame when no camera is available."""
        from PIL import ImageFont
        
        # Create a gradient background
        width, height = 640, 480
        test_img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # Create a green gradient background
        for y in range(height):
            # Calculate gradient color (dark green to light green)
            green = int(100 + (150 * y / height))
            color = (230, green, 230)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add text
        font_size = 24
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((width//2-150, height//2-50), "Test Mode - No Camera", 
                 fill="black", font=font)
        draw.text((width//2-150, height//2+20), "Photo Booth Ready", 
                 fill="#007F3F", font=font)
        
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
        
        # Convert PIL Image to array and then to BGR for OpenCV compatibility
        frame_array = np.array(test_img)
        self.current_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Store the test image - use direct approach to avoid cv2.cvtColor issue
        if hasattr(self, 'preview_canvas'):
            self._update_preview_ui_direct(test_img)
    
    def _update_preview_ui_direct(self, pil_image):
        """Update preview UI directly with a PIL image (bypassing OpenCV conversion)."""
        try:
                        # ✅ Convert PIL to OpenCV BGR and store as last frame
            frame_rgb = np.array(pil_image)  # RGB
            self.last_preview_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if hasattr(self, 'preview_canvas'):
                pil_image = self.preview_canvas.resize_to_fit(pil_image)
                photo_image = ImageTk.PhotoImage(image=pil_image)
                self.preview_canvas.photo_image = photo_image

                if hasattr(self.preview_canvas, 'placeholder_id') and self.preview_canvas.placeholder_id:
                    self.preview_canvas.delete(self.preview_canvas.placeholder_id)
                    self.preview_canvas.placeholder_id = None

                if hasattr(self.preview_canvas, 'image_id'):
                    self.preview_canvas.itemconfig(self.preview_canvas.image_id, image=photo_image)
                else:
                    self.preview_canvas.image_id = self.preview_canvas.create_image(
                    self.preview_canvas.width // 2,
                    self.preview_canvas.height // 2,
                    image=photo_image
                    )
                # Log success
                self._log_status("Preview updated with test image")
        except Exception as e:
            self._log_status(f"Error updating preview: {str(e)}")
            print(f"Error in _update_preview_ui_direct: {e}")
            traceback.print_exc()
    

# In GUI.py - Update _on_frame_received
    def _on_frame_received(self, frame):
        """Callback for frame processing with load shedding"""
        try:
            if frame is None or self.is_preview_frozen:
                return
    
            # Convert to RGB and resize before queuing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (320, 240))
            
            # Non-blocking queue put with frame dropping
            try:
                self.preview_queue.put_nowait(frame_small)
                self.after_idle(self._update_preview_ui)
            except queue.Full:
                pass  # Skip frame to prevent blocking
                
        except Exception as e:
            print(f"Frame handling error: {str(e)}")


    def _update_preview_ui(self):
        try:
            frame = self.preview_queue.get_nowait()
            self.preview_canvas.update_preview(frame)
            self.after(33, self._update_preview_ui)  # Throttle to ~30 FPS
        except queue.Empty:
            #self.after(10, self._update_preview_ui)  # Retry sooner if empty
            pass
        except Exception as e:
            print(f"Preview update error: {str(e)}")
            traceback.print_exc()
    
    def _on_capture_click(self):
        if self.is_processing:
            return  # prevent re-entry

        if self.latest_result:
        # Photo already taken → Reset
            self._reset_state()
            return
        name = datetime.now().strftime("photo_%Y%m%d_%H%M%S")
        frame = getattr(self, 'last_preview_frame', None)
        if frame is None:
            self._log_status("❌ No frame available to capture.")
            self.capture_btn.set_state("normal")

            self.is_processing = False
            return
        self.booth.set_user_name(name)
        self.capture_btn.set_state("disabled")
        

        self._start_countdown()

    def _start_countdown(self):
        self.countdown_seconds = 10
        self.is_counting_down = True
        self.is_preview_frozen = False  # Keep showing live preview


        self.countdown_btn.itemconfig(self.countdown_text_id, state="normal")

        self._update_countdown()

    def _update_countdown(self):
        if self.countdown_seconds > 0:
            self.countdown_btn.itemconfig(self.countdown_text_id, text=str(self.countdown_seconds))
            self.countdown_btn.itemconfig(self.countdown_text_id, state="normal")
            self.countdown_after_id = self.after(1000, self._update_countdown)
            self.countdown_seconds -= 1
        else:
            self.countdown_btn.itemconfig(self.countdown_text_id, state="hidden")
            self.is_counting_down = False
            self.is_preview_frozen = True
            self._trigger_photo_capture()



    def _trigger_photo_capture(self):
        frame = self.last_preview_frame
        if frame is None:
            self._log_status("❌ No frame available to capture.")
            self.capture_btn.set_state("normal")
            
            return

        self.is_processing = True
        self._log_status("📸 Capturing your photo...")

        self.thread_pool.submit(
        command_type="capture",
        data=lambda: self.booth.process_photo(frame),
        callback=self._on_process_complete
        )


    def _show_qr_code(self, qr_path):
        try:
            qr_img = Image.open(qr_path)
            qr_img = qr_img.resize((150, 150), Image.Resampling.LANCZOS)
            qr_photo = ImageTk.PhotoImage(qr_img)

            # Delete everything before drawing
            self.qr_canvas.delete("all")

            # Draw image on canvas centered
            canvas_width = self.qr_canvas.winfo_width()
            canvas_height = self.qr_canvas.winfo_height()

            self.qr_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=qr_photo
            )

            # ✅ Strong persistent reference to avoid garbage collection
            self.qr_image_tk = qr_photo

            self._log_status("📎 QR Code displayed.")

        except Exception as e:
            self._log_status(f"⚠ Failed to load QR code: {e}")
            print(f"Error displaying QR code: {e}")



    
    def _on_process_complete(self, result):
        self.is_processing = False
        self.latest_result = result
        self.is_preview_frozen = False

        if result is None:
            self._log_status("❌ Failed to process image.")
            self.capture_btn.set_state("normal")
            
            return

        self._log_status("✅ Processing complete. Tap 'Take Photo' to try again.")
        self.capture_btn.set_state("normal")

        qr_path = result.get("qr_code")
        if qr_path:
            self._show_qr_code(qr_path)
       


    def _reset_state(self):
        self.latest_result = None
        self.preview_canvas.clear()
        self.qr_canvas.delete("all")
        self.name_entry.entry.delete(0, tk.END)
        self.name_entry._put_placeholder()
        self
        self._log_status("Ready for a new photo.")

    def _on_process_click(self):
        """Handle process button click."""
        if self.current_frame is None:
            self._log_status("No photo to process")
            return
        
        # Get user name from entry field
        user_name = self.name_entry.get()
        if user_name:
            self.booth.set_user_name(user_name)
        
        # Disable buttons during processing
        self._set_buttons_state("process", "disabled")
        self.is_processing = True
        
        # Update status
        self._log_status("Processing photo...")
       
        
        # Define progress callback
        def progress_update(progress_data):
            current, total, percentage, message = progress_data
            
        
        # Process in a worker thread
        def process_callback(result):
            self.is_processing = False
            
            if result:
                self.latest_result = result
                self._log_status(f"Processing complete! Photo saved and QR code generated.")
                
                if isinstance(result, dict):
                    # Show the processed image if available
                    if 'processed' in result and result['processed']:
                        try:
                            processed_img = Image.open(result['processed'])
                            self.preview_canvas.display_photo(processed_img)
                            self._log_status(f"Cartoonified image displayed")
                        except Exception as e:
                            print(f"Error displaying processed image: {e}")
                            traceback.print_exc()
                    
                    # Show the QR code if available
                    if 'qr_code' in result and result['qr_code']:
                        try:
                            # Clear placeholder text
                            if hasattr(self, 'qr_placeholder_id'):
                                self.qr_canvas.delete(self.qr_placeholder_id)
                            
                            # Load QR code image
                            # Load QR code image from filepath
                            try:
                                qr_path = result['qr_code']
                                print(f"Loading QR code from: {qr_path}")
                                qr_img = Image.open(qr_path)
                            except Exception as e:
                                print(f"Error loading QR code: {e}")
                                raise
                            
                            # Make QR code much larger (300px out of 330px canvas to almost fill it)
                            qr_img = qr_img.resize((300, 300), resample=Image.Resampling.LANCZOS)
                            
                            # Convert to PhotoImage
                            self.qr_photo = ImageTk.PhotoImage(qr_img)
                            
                            # Clear any existing content on canvas including decorative elements
                            self.qr_canvas.delete("all")
                            
                            # Display on canvas centered (165,165 is center of 330x330 canvas)
                            self.qr_canvas.create_image(165, 165, image=self.qr_photo)
                            
                            self._log_status(f"QR code ready to scan")
                        except Exception as e:
                            print(f"Error displaying QR code: {e}")
                            traceback.print_exc()
                    
                    # Show URL if available
                    if 'url' in result and result['url']:
                        try:
                            # Update URL text
                            self.url_text.config(state=tk.NORMAL)
                            self.url_text.delete(1.0, tk.END)
                            self.url_text.insert(tk.END, result['url'])
                            self.url_text.config(state=tk.DISABLED)
                            
                            self._log_status(f"Share URL displayed")
                        except Exception as e:
                            print(f"Error displaying URL: {e}")
                            traceback.print_exc()
                
                
            else:
                self._log_status("Processing failed")
                
            
            # Re-enable buttons
            self._set_buttons_state("process", "normal")
        
        # Use the current frame for processing
        frame_copy = self.current_frame.copy()  # Make a copy to avoid thread issues
        
        # Submit the task
        self.thread_pool.submit(
            command_type="process",
            data=lambda: self.booth.process_photo(frame_copy, progress_update),
            callback=process_callback
        )
    
    def _perform_reset(self):
        """Perform the actual reset operation."""
        # Reset state variables
        self.is_processing = False
        self.latest_result = None
        
        # Clear the preview canvas
        self.preview_canvas.clear()
        
        # Reset progress bar
        
        
        # Re-enable all buttons
        self._set_buttons_state("all", "normal")
        
        # Clear status
        self._log_status("Reset complete. Ready for a new photo.")
        
        # Restart camera preview if needed
        if not self.camera_thread or not self.camera_thread.is_running():
            self._start_camera_preview()
    
    def _set_buttons_state(self, mode, state):
        """Set the state of buttons based on the current mode."""
        if mode == "capture" or mode == "all":
            self.capture_btn.set_state(state)
        
        if mode == "reset" or mode == "all":
            self.reset_btn.set_state(state)
    
    def _log_status(self, message):
        """Update the status message."""
        timestamp = time.strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}"
        
        # Update the status label
        self.status_label.config(text=status_message)
    
    def _on_closing(self):
        """Handle window closing event."""
        # Set a flag to stop threads
        if hasattr(self.booth, 'camera'):
            try:
                self.booth.camera.release()
            except:
                pass

        print("Closing application and cleaning up...")
        self._is_closing = True
        
        # Stop simulation if running
        if hasattr(self, 'simulation_active') and self.simulation_active:
            self.simulation_active = False
            print("Stopping simulation thread...")
            # Give simulation thread time to exit cleanly
            time.sleep(0.2)
        
        # Stop the camera thread
        if self.camera_thread:
            self.camera_thread.stop()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Destroy the window
        self.destroy()
    
    def mainloop(self):
        """Override mainloop to catch exceptions."""
        try:
            super().mainloop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
        finally:
            # Make sure we clean up
            if self.camera_thread:
                self.camera_thread.stop()
            self.thread_pool.shutdown(wait=True)




class ImageButton(tk.Canvas):
    def __init__(self, parent, image_path, command, size=(64, 64)):
        super().__init__(parent, width=size[0], height=size[1], highlightthickness=0, bg="#F0F0F0")

        self.command = command
        self.state = "normal"

        # Load original image
        img = Image.open(image_path).convert("RGBA")

        # Auto-crop the image to its content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Resize to target button size
        img = img.resize(size, Image.Resampling.LANCZOS)

        self.image_normal = ImageTk.PhotoImage(img)

        # Create dimmed version
        img_dim = img.copy().convert("RGBA")
        pixels = img_dim.getdata()
        dimmed_pixels = []

        for r, g, b, a in pixels:
            if a == 0:
                dimmed_pixels.append((0, 0, 0, 0))  # fully transparent remains transparent
            else:
                dimmed_pixels.append((int(r * 0.6), int(g * 0.6), int(b * 0.6), a))

        img_dim.putdata(dimmed_pixels)
        self.image_dim = ImageTk.PhotoImage(img_dim)

        # Create pressed version
        img_pressed = img.copy().point(lambda p: p * 0.8)
        self.image_pressed = ImageTk.PhotoImage(img_pressed)

        self.current_image = self.create_image(size[0] // 2, size[1] // 2, image=self.image_normal)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, event):
        if self.state == "normal":
            self.itemconfig(self.current_image, image=self.image_normal)

    def _on_leave(self, event):
        if self.state == "normal":
            self.itemconfig(self.current_image, image=self.image_normal)

    def _on_click(self, event):
        if self.state == "normal":
            self.itemconfig(self.current_image, image=self.image_pressed)

    def _on_release(self, event):
        if self.state == "normal":
            self.itemconfig(self.current_image, image=self.image_normal)
            if self.command:
                self.command()

    def set_state(self, state):
        self.state = state
        if state == "disabled":
            self.itemconfig(self.current_image, image=self.image_dim)
            self.unbind("<Enter>")
            self.unbind("<Leave>")
            self.unbind("<Button-1>")
            self.unbind("<ButtonRelease-1>")
        else:
            self.itemconfig(self.current_image, image=self.image_normal)
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<Button-1>", self._on_click)
            self.bind("<ButtonRelease-1>", self._on_release)
