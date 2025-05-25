import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import messagebox

# Add the attached_assets directory to Python path to find modules
attached_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attached_assets')
if attached_assets_dir not in sys.path:
    sys.path.append(attached_assets_dir)

# Import modules from attached_assets
from backend import PhotoBooth
from GUI import PhotoBoothGUI


def setup_required_directories():
    """Setup required directories for the photo booth."""
    # Base directories
    for directory in ["photos", "processed", "qrcodes"]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory '{directory}' is ready")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")


def main():
    try:
        # Ensure required directories exist
        setup_required_directories()
        
        # Create backend instance
        print("Initializing photo booth backend...")
        booth = PhotoBooth()
        
        # Print active threads (for debugging)
        print(f"Initial active threads: {threading.active_count()}")
        
        # Create GUI application
        print("Starting GUI application...")
        app = PhotoBoothGUI(booth)
        
        # Enable test mode if camera initialization failed
        #print("Checking camera availability...")
        #result = booth.initialize_camera()
        #if result is None:
            #print("No camera detected - enabling test mode")
            #app._start_simulation_mode()
        
        # Start main loop
        app.mainloop()
        
        # Print final thread count (for debugging)
        print(f"Final active threads: {threading.active_count()}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        
        # In case of error, try to display a simple error message
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", 
                                f"An error occurred:\n{str(e)}\n\nCheck the console for details.")
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
