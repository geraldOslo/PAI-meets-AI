#!/usr/bin/env python3
"""
********************************************************************************
* Dental X-Ray Periapical Index (PAI) Analyzer
********************************************************************************
*
* Version: 2.0.0
* Author: Gerald Torgersen 2025
* GitHub: https://github.uio.no/gerald/PAI-meets-AI
*
* Description:
* This application provides automated analysis of dental radiographs using 
* deep learning to classify periapical lesions according to the Periapical 
* Index (PAI) system. The PAI scale ranges from 1 (healthy) to 5 (severe lesion).
*
* Features:
* - Support for new checkpoint format with automatic model configuration loading
* - Multi-architecture support: ResNet50, EfficientNet-B3, ConvNeXt-Tiny
* - Automatic detection of optimal input size per architecture (224x224 or 300x300)
* - Support for TIFF and standard image formats
* - Grad-CAM visualization for explainable AI
* - Adjustable pixel size for different image resolutions
* - Region-of-interest extraction with consistent physical dimensions
* - Multi-threading for responsive UI during analysis
* - Automatic CPU/GPU detection and optimization
*
* Usage:
* 1. Run the application
* 2. Click "Load Model" button and select a model checkpoint (.pth file)
*    OR specify checkpoint path via command-line: --checkpoint path/to/model.pth
*    OR place checkpoint in './model/' subdirectory for auto-detection
* 3. Click "Load Image" and select a dental radiograph
* 4. Set the correct pixel size in mm (typically 0.03-0.05mm for dental radiographs)
* 5. Click on a point of interest in the image
* 6. View the PAI classification results and Grad-CAM visualizations
*
* Directory Structure:
* project_root/
*   ├── code/
*   │    └── inference/
*   │         ├── dental_xray_analyzer.py    # This script
*   │         └── model/                     # Subdirectory for model checkpoint
*   │              └── your_checkpoint.pth  # Model checkpoint (not included in repo)
*
* Requirements:
* - Python 3.7+
* - PyTorch 1.8+
* - timm library for EfficientNet models
* - OpenCV, NumPy, Matplotlib
* - Tkinter for GUI
*
* Model Details:
* - Architectures: ResNet50, EfficientNet-B3, or ConvNeXt-Tiny (loaded via timm)
* - Input size: Auto-detected (224×224 for ResNet/ConvNeXt, 300×300 for EfficientNet)
* - Region size: 12×12 mm at reference pixel size of 0.04mm
* - Output: 5 classes (PAI 1-5)
* - Model configuration automatically loaded from checkpoint
*
********************************************************************************
"""

import os
import numpy as np
import sys
import traceback
from pathlib import Path

# Import torch first to check for CUDA
import torch

# Try to import timm (required for new model format)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("ERROR: timm library not found. Please install it: pip install timm")
    sys.exit(1)

# Set device with proper error handling
def setup_device():
    """Setup the best available device (CUDA or CPU) with proper error handling."""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU for computation")
    except Exception as e:
        device = torch.device('cpu')
        print(f"Error checking CUDA availability: {e}")
        print("Defaulting to CPU")
    return device

# Initialize device
device = setup_device()

# Import remaining modules
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, ttk, Checkbutton, IntVar, DoubleVar, Entry, messagebox
import threading
import queue
import argparse

# Default normalization constants (will be updated from checkpoint if available)
DEFAULT_MEAN = [0.3784975, 0.3784975, 0.3784975]
DEFAULT_STD = [0.16739018, 0.16739018, 0.16739018]

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    This allows the application to find resources whether run directly
    or from a bundled executable.
    
    Args:
        relative_path: Path relative to the script or executable
        
    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    
    return os.path.join(base_path, relative_path)

def find_model_checkpoint(model_dir):
    """
    Find the first .pth checkpoint file in the model directory.
    
    Args:
        model_dir: Directory to search for checkpoint files
        
    Returns:
        Path to checkpoint file or None if not found
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return None
    
    # Look for .pth files
    checkpoint_files = list(model_path.glob("*.pth"))
    if checkpoint_files:
        return str(checkpoint_files[0])  # Return the first one found
    
    return None

def get_model_input_size(model_name):
    """
    Get the optimal input size for a given model architecture.

    Args:
        model_name: Name of the model (e.g., 'efficientnet_b3', 'resnet50')

    Returns:
        Input size in pixels (int)
    """
    model_name_lower = model_name.lower()

    if 'efficientnet' in model_name_lower:
        return 300  # EfficientNet-B3 uses 300x300
    elif 'resnet' in model_name_lower:
        return 224  # ResNet50 uses 224x224
    elif 'convnext' in model_name_lower:
        return 224  # ConvNeXt-Tiny uses 224x224
    else:
        # Default to 224 for unknown architectures
        print(f"Warning: Unknown architecture '{model_name}', defaulting to 224x224")
        return 224

def infer_model_from_filename(checkpoint_path):
    """
    Infer model architecture from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with model configuration or None if cannot infer
    """
    filename = os.path.basename(checkpoint_path).lower()

    # Model mappings based on common naming patterns
    if 'efficientnet-b3' in filename or 'efficientnet_b3' in filename or 'effnet' in filename:
        return {
            'model_name': 'efficientnet_b3',
            'timm_name': 'efficientnet_b3',
            'input_size': 300,
            'dropout_rate': 0.5,
            'drop_path_rate': 0.2,
            'num_classes': 5
        }
    elif 'resnet50' in filename or 'resnet_50' in filename or 'resnet' in filename:
        return {
            'model_name': 'resnet50',
            'timm_name': 'resnet50',
            'input_size': 224,
            'dropout_rate': 0.5,
            'drop_path_rate': 0.1,
            'num_classes': 5
        }
    elif 'convnext' in filename:
        return {
            'model_name': 'convnext_tiny',
            'timm_name': 'convnext_tiny',
            'input_size': 224,
            'dropout_rate': 0.5,
            'drop_path_rate': 0.2,
            'num_classes': 5
        }

    return None

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint with support for both new and old formats.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, normalization_mean, normalization_std, target_layer, input_size, model_config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint file: {e}")

    # Try to extract model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})

    # If no model_config found, this is an old format checkpoint (state_dict only)
    is_old_format = not model_config

    if is_old_format:
        print("Old format checkpoint detected (state_dict only)")
        print("Attempting to infer model architecture from filename...")

        # Try to infer from filename
        inferred_config = infer_model_from_filename(checkpoint_path)

        if not inferred_config:
            raise ValueError(
                f"Cannot automatically determine model architecture from filename: {os.path.basename(checkpoint_path)}\n"
                f"Please ensure the checkpoint filename contains one of: 'resnet50', 'efficientnet_b3', or 'convnext'"
            )

        model_config = inferred_config
        print(f"Inferred model architecture: {model_config['model_name']}")
    else:
        print("New format checkpoint detected (with metadata)")

    print("Model configuration:")
    print(f"  Model name: {model_config.get('model_name')}")
    print(f"  Num classes: {model_config.get('num_classes')}")
    print(f"  Dropout rate: {model_config.get('dropout_rate')}")
    print(f"  Drop path rate: {model_config.get('drop_path_rate')}")
    
    # Extract configuration parameters
    model_name = model_config.get('model_name')
    num_classes = model_config.get('num_classes', 5)
    dropout_rate = model_config.get('dropout_rate', 0.0)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)
    
    # Validate required parameters
    if not model_name:
        raise ValueError("model_name not found in checkpoint's model_config")
    
    print(f"Creating model architecture '{model_name}' with {num_classes} classes...")
    
    # Create model using timm
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,  # Don't load ImageNet weights
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate
        )
        print("Model architecture created successfully")
    except Exception as e:
        raise RuntimeError(f"Error creating model architecture '{model_name}': {e}")

    # Load model weights
    print("Loading model weights...")
    try:
        # Handle different checkpoint formats
        if is_old_format:
            # Old format: checkpoint IS the state_dict
            model_state_dict = checkpoint
            print("  Using checkpoint as state_dict (old format)")
        else:
            # New format: checkpoint contains model_state_dict key
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(model_state_dict, dict) and 'state_dict' in model_state_dict:
                model_state_dict = model_state_dict['state_dict']

        # Handle DataParallel prefix if present
        if model_state_dict and list(model_state_dict.keys())[0].startswith('module.'):
            print("  Removing 'module.' prefix from state_dict keys")
            model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}

        # Load weights
        model.load_state_dict(model_state_dict, strict=True)
        print("Model weights loaded successfully")

    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Try with strict=False as fallback
        try:
            model.load_state_dict(model_state_dict, strict=False)
            print("Model weights loaded with strict=False (some weights may be missing)")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model weights: {e2}")

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Optimize for inference
    if device.type == 'cpu':
        print("Optimizing model for CPU inference...")
        model = model.float()  # Ensure float32 precision for CPU
        # Limit threads to avoid overwhelming the system
        if torch.get_num_threads() > 4:
            torch.set_num_threads(min(4, os.cpu_count() - 1))
    else:
        print("Optimizing model for GPU inference...")
        # Enable optimizations for GPU
        torch.backends.cudnn.benchmark = True

    # Get target layer for Grad-CAM (last feature layer)
    target_layer = None
    try:
        # For EfficientNet in timm, the feature extractor is typically model.features
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif hasattr(model, 'blocks'):
            target_layer = model.blocks[-1]
        else:
            # Fallback: try to find the last convolutional layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    target_layer = module
                    break
        
        if target_layer is None:
            print("Warning: Could not automatically identify target layer for Grad-CAM")
        else:
            print(f"Target layer for Grad-CAM: {target_layer}")
            
    except Exception as e:
        print(f"Warning: Error identifying target layer: {e}")
        target_layer = None

    # Extract normalization parameters if available
    norm_params = checkpoint.get('normalization', {})
    mean = norm_params.get('mean', DEFAULT_MEAN)
    std = norm_params.get('std', DEFAULT_STD)

    print(f"Using normalization - Mean: {mean}, Std: {std}")

    # Determine optimal input size for this architecture
    input_size = get_model_input_size(model_name)
    print(f"Using input size: {input_size}x{input_size} pixels")

    # Print checkpoint info if available
    if 'epoch' in checkpoint:
        print(f"Checkpoint info: Epoch {checkpoint['epoch']}")
    if 'best_metric_val' in checkpoint:
        print(f"Best validation metric: {checkpoint['best_metric_val']}")
    if 'timestamp' in checkpoint:
        print(f"Training timestamp: {checkpoint['timestamp']}")

    return model, mean, std, target_layer, input_size, model_config

class XRayAnalyzerApp:
    """
    Main application class for the Dental X-Ray Analyzer.
    Handles the GUI, image processing, and model inference with the new checkpoint format.
    """
    
    def __init__(self, root, checkpoint_path=None):
        """
        Initialize the application.

        Args:
            root: The tkinter root window
            checkpoint_path: Optional path to specific checkpoint file
        """
        self.root = root
        self.root.title("Dental X-Ray PAI Analyzer v2.0 (Multi-Architecture)")
        self.checkpoint_path = checkpoint_path
        self.root.geometry("1200x800")

        # Application state
        self.device = device
        self.status_queue = queue.Queue()
        self.analysis_running = False

        # Model and normalization parameters (loaded from checkpoint)
        self.model = None
        self.model_name = None  # Store model name for display
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.target_layer = None
        self.input_size = 300  # Default, will be updated from checkpoint

        # Image variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.current_point = None

        # Analysis options
        self.show_gradcam = IntVar(value=1)  # Default to showing Grad-CAM

        # Pixel size settings - crucial for accurate physical measurements
        self.reference_pixel_size = 0.04  # mm - reference pixel size for model training
        self.image_pixel_size = DoubleVar(value=self.reference_pixel_size)

        # GUI setup
        self.setup_gui()

        # Start status update timer
        self.check_status_queue()

        # Only auto-load model if checkpoint path was provided via command-line
        if checkpoint_path:
            self.status_label.config(text="Loading model... Please wait.")
            self.loading_thread = threading.Thread(target=self.load_model_thread)
            self.loading_thread.daemon = True
            self.loading_thread.start()
        else:
            self.status_label.config(text="Ready! Load a model to begin.")
        
    def load_model_dialog(self):
        """Open file dialog to select model checkpoint."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[
                ("PyTorch checkpoints", "*.pth"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            # Update checkpoint path
            self.checkpoint_path = file_path

            # Disable load model button during loading
            self.load_model_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Loading model... Please wait.")

            # Start loading model in background
            loading_thread = threading.Thread(target=self.load_model_thread)
            loading_thread.daemon = True
            loading_thread.start()

    def load_model_thread(self):
        """Load model in background thread to keep UI responsive."""
        try:
            # Determine checkpoint path
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                # Use specified checkpoint path
                checkpoint_path = self.checkpoint_path
                self.status_queue.put(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
            else:
                # Look for checkpoint in 'model' subdirectory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(script_dir, 'model')

                self.status_queue.put("Searching for model checkpoint...")

                # Find checkpoint file
                checkpoint_path = find_model_checkpoint(model_dir)

                if not checkpoint_path:
                    raise FileNotFoundError(f"No .pth checkpoint files found in {model_dir}")

                self.status_queue.put(f"Found checkpoint: {os.path.basename(checkpoint_path)}")

            # Load model using new checkpoint format
            self.status_queue.put("Loading model architecture and weights...")
            model, mean, std, target_layer, input_size, model_config = load_model_from_checkpoint(checkpoint_path, self.device)

            # Store model and parameters
            self.model = model
            self.model_name = model_config.get('model_name', 'Unknown')
            self.mean = mean
            self.std = std
            self.target_layer = target_layer
            self.input_size = input_size

            self.status_queue.put("Model loaded successfully!")

            # Enable UI elements in the main thread
            self.root.after(0, self.enable_ui_after_loading)

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.status_queue.put(error_msg)

            # Re-enable load model button on error
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL))

            # Show error dialog in main thread
            self.root.after(0, lambda: messagebox.showerror("Model Loading Error", error_msg))
    
    def enable_ui_after_loading(self):
        """Enable UI elements after model loading completes."""
        self.load_image_btn.config(state=tk.NORMAL)
        self.load_model_btn.config(state=tk.NORMAL)

        # Update model info display
        if self.model_name:
            model_display = self.model_name.replace('_', ' ').title()
            self.model_info_label.config(text=f"Model: {model_display} ({self.input_size}×{self.input_size})",
                                        fg="darkgreen")

        self.status_label.config(text="Ready! Load an image and click on an apex to analyze.")
    
    def check_status_queue(self):
        """Check for status updates from background threads."""
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get(block=False)
                self.status_label.config(text=message)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_status_queue)
    
    def setup_gui(self):
        """Set up the GUI components with improved layout."""
        # Main frames for layout
        self.left_frame = Frame(self.root, width=600, height=800)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = Frame(self.root, width=600, height=800)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_frame.pack_propagate(False)
        
        # Left frame - Image display and controls
        self.setup_controls()
        self.setup_image_display()
        
        # Right frame - Analysis results
        self.setup_results_display()
    
    def setup_controls(self):
        """Set up the control panel."""
        self.controls_frame = Frame(self.left_frame)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Top row of controls
        top_controls = Frame(self.controls_frame)
        top_controls.pack(fill=tk.X, pady=(0, 5))

        # Load model button
        self.load_model_btn = Button(top_controls, text="Load Model", command=self.load_model_dialog,
                                     font=("Arial", 10, "bold"), bg="lightblue")
        self.load_model_btn.pack(side=tk.LEFT, padx=5)

        # Load image button - initially disabled until model loads
        self.load_image_btn = Button(top_controls, text="Load Image", command=self.load_image,
                                     state=tk.DISABLED, font=("Arial", 10, "bold"))
        self.load_image_btn.pack(side=tk.LEFT, padx=5)

        # Grad-CAM visualization toggle
        self.gradcam_check = Checkbutton(top_controls, text="Show Grad-CAM",
                                        variable=self.show_gradcam, font=("Arial", 10))
        self.gradcam_check.pack(side=tk.LEFT, padx=10)

        # Quit button
        self.quit_btn = Button(top_controls, text="Quit", command=self.root.quit,
                              font=("Arial", 10))
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
        
        # Bottom row - pixel size input
        bottom_controls = Frame(self.controls_frame)
        bottom_controls.pack(fill=tk.X)
        
        # Pixel size input - critical for accurate physical measurements
        Label(bottom_controls, text="Image Pixel Size (mm):", font=("Arial", 10)).pack(side=tk.LEFT)
        
        pixel_size_entry = Entry(bottom_controls, textvariable=self.image_pixel_size, 
                                width=8, font=("Arial", 10))
        pixel_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Help text for pixel size
        help_text = "Typical range: 0.03-0.05mm"
        Label(bottom_controls, text=help_text, font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=5)
        
        # Model info label (will be updated when model loads)
        self.model_info_label = Label(bottom_controls, text="No model loaded",
                                      font=("Arial", 9, "bold"), fg="darkred")
        self.model_info_label.pack(side=tk.LEFT, padx=15)

        # Device info
        device_info = f"Device: {self.device.type.upper()}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name()})"
        Label(bottom_controls, text=device_info, font=("Arial", 8), fg="blue").pack(side=tk.RIGHT, padx=5)
    
    def setup_image_display(self):
        """Set up the image display area."""
        # Instructions for the user
        self.instr_label = Label(self.left_frame, 
                                text="Click on a tooth apex to analyze a 12×12 mm region",
                                font=("Arial", 11), fg="darkblue")
        self.instr_label.pack(pady=5)
        
        # Canvas for displaying the radiograph
        self.image_canvas = tk.Canvas(self.left_frame, bg="black", relief=tk.SUNKEN, bd=2)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_canvas.bind("<Button-1>", self.on_image_click)
        
        # Progress bar for long operations
        self.progress_frame = Frame(self.left_frame)
        self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                           length=100, mode='indeterminate')
        
        # Status bar for user feedback
        self.status_label = Label(self.left_frame, text="Starting application...", 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_results_display(self):
        """Set up the results display area."""
        # Results header
        self.results_label = Label(self.right_frame, text="Results will appear here", 
                                  font=("Arial", 14, "bold"))
        self.results_label.pack(pady=10)
        
        # Create a frame for matplotlib visualizations
        self.fig_frame = Frame(self.right_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup matplotlib figure
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add instructions
        instructions = Label(self.right_frame, 
                           text="Load an image and click on a tooth apex for PAI analysis",
                           font=("Arial", 10), fg="gray", wraplength=300)
        instructions.pack(pady=5)
    
    def load_image(self):
        """Load and display a dental radiograph."""
        # Check if model is loaded
        if self.model is None:
            self.status_label.config(text="Model still loading. Please wait...")
            return
            
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Dental Radiograph",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_label.config(text=f"Loading: {os.path.basename(file_path)}")
                
                # Load and convert image
                self.image_path = file_path
                self.original_image = Image.open(file_path).convert('RGB')
                self.display_image = self.original_image.copy()
                
                # Update display
                self.update_image_canvas()
                
                # Update UI
                filename = os.path.basename(file_path)
                self.instr_label.config(text=f"Loaded: {filename} - Click on a tooth apex to analyze")
                self.status_label.config(text=f"Ready - Image loaded: {filename}")
                
                # Clear previous results
                self.clear_results()
                
            except Exception as e:
                error_msg = f"Error loading image: {str(e)}"
                self.status_label.config(text=error_msg)
                messagebox.showerror("Image Loading Error", error_msg)
    
    def clear_results(self):
        """Clear previous analysis results."""
        self.fig.clear()
        self.canvas.draw()
        self.results_label.config(text="Results will appear here")
    
    def update_image_canvas(self):
        """Display the loaded image on the canvas, properly scaled."""
        if not self.display_image:
            return
            
        # Clear canvas
        self.image_canvas.delete("all")
        
        # Get canvas dimensions
        self.image_canvas.update()  # Ensure canvas is drawn
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Use reasonable defaults if canvas not yet drawn
        if canvas_width <= 1:
            canvas_width = 580
        if canvas_height <= 1:
            canvas_height = 580
        
        # Calculate scaling to fit image in canvas while maintaining aspect ratio
        img_width, img_height = self.display_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)  # Don't upscale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image for display
        self.display_scaled = self.display_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_scaled)
        
        # Display image centered on canvas
        self.image_x = (canvas_width - new_width) // 2
        self.image_y = (canvas_height - new_height) // 2
        self.image_canvas.create_image(self.image_x, self.image_y, 
                                      anchor=tk.NW, image=self.tk_image)
        
        # Store scaling factor for coordinate conversion
        self.scale_factor = img_width / new_width
    
    def on_image_click(self, event):
        """Handle click on the image canvas to initiate analysis."""
        if (self.display_image and self.model is not None and 
            not self.analysis_running and hasattr(self, 'display_scaled')):
            
            # Convert canvas coordinates to displayed image coordinates
            canvas_x = event.x - self.image_x
            canvas_y = event.y - self.image_y
            
            # Check if click is inside the image
            if (0 <= canvas_x < self.display_scaled.width and 
                0 <= canvas_y < self.display_scaled.height):
                
                # Convert to original image coordinates
                orig_x = int(canvas_x * self.scale_factor)
                orig_y = int(canvas_y * self.scale_factor)
                
                # Store click point and start analysis
                self.current_point = (orig_x, orig_y)
                self.start_analysis_thread()
    
    def start_analysis_thread(self):
        """Start the analysis process in a background thread."""
        # Set analysis state
        self.analysis_running = True
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar.start(10)
        self.status_label.config(text="Analyzing region... Please wait.")
        self.load_image_btn.config(state=tk.DISABLED)
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def run_analysis(self):
        """Run the PAI analysis in a background thread."""
        try:
            # Extract region
            self.status_queue.put("Extracting 12×12 mm region...")
            region = self.extract_region(self.current_point[0], self.current_point[1])
            
            # Preprocess region
            self.status_queue.put("Preprocessing image...")
            preprocess = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            input_tensor = preprocess(region).unsqueeze(0).to(self.device)
            
            # Run inference
            self.status_queue.put("Running PAI classification...")
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output[0], dim=0).cpu().numpy()
            
            predicted_class = np.argmax(probabilities) + 1  # Convert to 1-based PAI
            
            # Update with initial results
            self.root.after(0, lambda: self.update_initial_results(region, predicted_class, probabilities))
            
            # Generate Grad-CAM if requested
            if self.show_gradcam.get() and self.target_layer is not None:
                self.status_queue.put("Generating Grad-CAM visualizations...")
                gradcam_heatmaps = []
                for target_class in range(5):
                    heatmap = self.get_gradcam(target_class, input_tensor)
                    gradcam_heatmaps.append(heatmap)
                
                # Update with Grad-CAM results
                self.status_queue.put("Analysis complete!")
                self.root.after(0, lambda: self.update_gradcam_results(region, predicted_class, probabilities, gradcam_heatmaps))
            else:
                self.status_queue.put("Analysis complete!")
                self.root.after(0, self.reset_analysis_ui)
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.status_queue.put(error_msg)
            self.root.after(0, self.reset_analysis_ui)
    
    def extract_region(self, center_x, center_y, target_size=None):
        """
        Extract a 12×12 mm region with consistent physical dimensions.

        Args:
            center_x, center_y: Center coordinates in pixels
            target_size: Target output size for the model (default: use self.input_size)

        Returns:
            PIL Image of the extracted region, properly padded and sized
        """
        # Use model-specific input size if not specified
        if target_size is None:
            target_size = self.input_size
        # Get image dimensions
        img_width, img_height = self.original_image.size
        
        # Calculate physical size (12×12 mm) in pixels for this image
        physical_size_mm = 12.0  # Always extract 12×12 mm
        physical_size_pixels = int(physical_size_mm / self.image_pixel_size.get())
        half_size = physical_size_pixels // 2
        
        # Calculate region boundaries
        left = max(0, center_x - half_size)
        top = max(0, center_y - half_size)
        right = min(img_width, center_x + half_size)
        bottom = min(img_height, center_y + half_size)
        
        # Create padded region with black background
        padded_region = Image.new('RGB', (physical_size_pixels, physical_size_pixels), (0, 0, 0))
        
        # Extract region from original image
        region = self.original_image.crop((left, top, right, bottom))
        
        # Calculate paste position to center the extracted region
        paste_x = (physical_size_pixels - region.width) // 2
        paste_y = (physical_size_pixels - region.height) // 2
        
        # Adjust for edge cases
        if center_x < half_size:
            paste_x = half_size - center_x
        if center_y < half_size:
            paste_y = half_size - center_y
        if center_x + half_size > img_width:
            paste_x = physical_size_pixels - region.width
        if center_y + half_size > img_height:
            paste_y = physical_size_pixels - region.height
        
        # Paste region onto padded background
        padded_region.paste(region, (max(0, paste_x), max(0, paste_y)))
        
        # Resize to model input size (300×300)
        result = padded_region.resize((target_size, target_size), Image.LANCZOS)
        
        return result
    
    def update_initial_results(self, region, predicted_class, probabilities):
        """Show initial prediction results."""
        # Update results label
        confidence = probabilities[predicted_class - 1] * 100
        self.results_label.config(text=f"Predicted PAI: {predicted_class} ({confidence:.1f}%)")
        
        # Clear previous figure
        self.fig.clear()
        
        # Create subplot layout
        ax_img = self.fig.add_axes([0.1, 0.55, 0.35, 0.35])
        ax_prob = self.fig.add_axes([0.55, 0.55, 0.35, 0.35])
        
        # Show extracted region
        ax_img.imshow(region)
        ax_img.set_title(f'Extracted Region\nPredicted PAI: {predicted_class}', fontsize=11)
        ax_img.set_xlabel(f'12×12 mm region\n(Pixel size: {self.image_pixel_size.get():.3f} mm)', fontsize=9)
        ax_img.axis('off')
        
        # Show probability distribution
        classes = np.arange(1, 6)
        colors = ['red' if i == predicted_class - 1 else 'lightblue' for i in range(5)]
        bars = ax_prob.bar(classes, probabilities, color=colors, edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax_prob.set_title('PAI Probability Distribution', fontsize=11)
        ax_prob.set_xlabel('PAI Class', fontsize=10)
        ax_prob.set_ylabel('Probability', fontsize=10)
        ax_prob.set_ylim(0, 1.1)
        ax_prob.set_xticks(classes)
        ax_prob.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Reset UI if Grad-CAM is disabled
        if not self.show_gradcam.get():
            self.reset_analysis_ui()
    
    def update_gradcam_results(self, region, predicted_class, probabilities, gradcam_heatmaps):
        """Update results with Grad-CAM visualizations."""
        # Reset UI
        self.reset_analysis_ui()
        
        # Clear figure
        self.fig.clear()
        
        # Top row: region and probabilities (same as before)
        ax_img = self.fig.add_axes([0.1, 0.55, 0.35, 0.35])
        ax_prob = self.fig.add_axes([0.55, 0.55, 0.35, 0.35])
        
        # Show extracted region
        ax_img.imshow(region)
        ax_img.set_title(f'Extracted Region\nPredicted PAI: {predicted_class}', fontsize=11)
        ax_img.set_xlabel(f'12×12 mm region\n(Pixel size: {self.image_pixel_size.get():.3f} mm)', fontsize=9)
        ax_img.axis('off')
        
        # Show probabilities
        classes = np.arange(1, 6)
        colors = ['red' if i == predicted_class - 1 else 'lightblue' for i in range(5)]
        bars = ax_prob.bar(classes, probabilities, color=colors, edgecolor='black', linewidth=1)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax_prob.set_title('PAI Probability Distribution', fontsize=11)
        ax_prob.set_xlabel('PAI Class', fontsize=10)
        ax_prob.set_ylabel('Probability', fontsize=10)
        ax_prob.set_ylim(0, 1.1)
        ax_prob.set_xticks(classes)
        ax_prob.grid(True, alpha=0.3)
        
        # Bottom row: Grad-CAM visualizations for all PAI classes
        width = 0.16
        spacing = 0.02
        y_pos = 0.08
        height = 0.35
        left_margin = 0.05
        
        region_array = np.array(region)
        
        for i in range(5):
            x_pos = left_margin + i * (width + spacing)
            ax = self.fig.add_axes([x_pos, y_pos, width, height])
            
            # Create overlaid visualization
            heatmap = gradcam_heatmaps[i]
            colored_heatmap = self.apply_colormap(heatmap)
            
            # Overlay heatmap on original region
            overlay = 0.6 * region_array / 255.0 + 0.4 * colored_heatmap / 255.0
            
            ax.imshow(overlay)
            
            # Highlight predicted class
            title_color = 'red' if i == predicted_class - 1 else 'black'
            title_weight = 'bold' if i == predicted_class - 1 else 'normal'
            ax.set_title(f'PAI {i+1}', fontsize=10, color=title_color, weight=title_weight)
            ax.axis('off')
        
        self.canvas.draw()
        
        # Update status
        confidence = probabilities[predicted_class - 1] * 100
        self.status_label.config(text=f"Analysis complete! Predicted PAI: {predicted_class} ({confidence:.1f}%)")
    
    def reset_analysis_ui(self):
        """Reset UI elements after analysis completes."""
        self.analysis_running = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.load_image_btn.config(state=tk.NORMAL)
    
    def get_gradcam(self, class_idx, input_tensor):
        """
        Generate Grad-CAM visualization for a specific class.

        Args:
            class_idx: Class index (0-4 for PAI 1-5)
            input_tensor: Preprocessed input tensor

        Returns:
            Normalized heatmap as numpy array
        """
        if self.target_layer is None:
            # Return empty heatmap if no target layer
            return np.zeros((self.input_size, self.input_size), dtype=np.float32)
        
        self.model.eval()

        # Hooks for capturing activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            # Clone output to avoid in-place operation issues
            activations.append(output.clone() if isinstance(output, torch.Tensor) else output)

        def backward_hook(module, grad_input, grad_output):
            # Clone gradient to avoid in-place operation issues
            if grad_output[0] is not None:
                gradients.append(grad_output[0].clone())

        # Register hooks
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            # Enable gradients for this operation
            input_tensor.requires_grad_(True)

            # Forward pass
            output = self.model(input_tensor)

            # Clear gradients
            self.model.zero_grad()

            # Backward pass for specified class
            target = output[0, class_idx]
            target.backward(retain_graph=False)

            # Get gradients and activations
            if not gradients:
                # If no gradients captured, return empty heatmap
                return np.zeros((self.input_size, self.input_size), dtype=np.float32)

            grads = gradients[0].cpu().data.numpy()
            acts = activations[0].cpu().data.numpy()

            # Compute Grad-CAM
            weights = np.mean(grads, axis=(2, 3))[0, :]
            grad_cam = np.zeros(acts.shape[2:], dtype=np.float32)

            # Weight activations by gradients
            for i, w in enumerate(weights):
                grad_cam += w * acts[0, i, :, :]

            # Apply ReLU and normalize
            grad_cam = np.maximum(grad_cam, 0)
            grad_cam = self.normalize_heatmap(grad_cam)

            # Resize to input size
            grad_cam = cv2.resize(grad_cam, (self.input_size, self.input_size))

            return grad_cam

        except Exception as e:
            print(f"Error in GradCAM computation: {e}")
            # Return empty heatmap on error
            return np.zeros((self.input_size, self.input_size), dtype=np.float32)

        finally:
            # Always clean up hooks
            handle_forward.remove()
            handle_backward.remove()
            # Clear gradients
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
    
    def normalize_heatmap(self, heatmap):
        """Normalize heatmap to range 0-1."""
        if np.max(heatmap) - np.min(heatmap) > 1e-8:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        return heatmap

    def apply_colormap(self, heatmap, cmap=cv2.COLORMAP_JET):
        """Apply colormap to heatmap."""
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cmap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        return colored_heatmap


def main():
    """Main application entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Dental X-Ray PAI Analyzer - Multi-Architecture Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model in ./model/ subdirectory
  python PAI_inference.py

  # Specify checkpoint for ResNet50
  python PAI_inference.py --checkpoint model_checkpoints/p5_resnet_alpha_0_5_20251028_113346/resnet50_best.pth

  # Specify checkpoint for EfficientNet-B3
  python PAI_inference.py --checkpoint model_checkpoints/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth

  # Specify checkpoint for ConvNeXt-Tiny
  python PAI_inference.py --checkpoint model_checkpoints/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth
        """
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint file (.pth)'
    )
    args = parser.parse_args()

    try:
        # Check for required libraries
        if not TIMM_AVAILABLE:
            print("ERROR: Required library 'timm' not found.")
            print("Please install it using: pip install timm")
            return

        # Create and run application
        root = tk.Tk()
        app = XRayAnalyzerApp(root, checkpoint_path=args.checkpoint)
        
        # Set minimum window size
        root.minsize(1000, 600)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (root.winfo_screenheight() // 2) - (800 // 2)
        root.geometry(f"1200x800+{x}+{y}")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        messagebox.showerror("Fatal Error", f"Application failed to start: {e}")


if __name__ == "__main__":
    main()