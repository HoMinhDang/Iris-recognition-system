import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

# Import the required modules
from pipeline import IrisRecognitionPipeline
from segmentation import WorldCoinSegmentator
from normalization import WorldCoinsNormalizer
from feature_extraction import WorldCoinsFeatureExtractor
from matching import WorldCoinsMatcher
from database import IrisDatabase

# Initialize the pipeline components
segmentator = WorldCoinSegmentator()
normalizer = WorldCoinsNormalizer()
feature_extractor = WorldCoinsFeatureExtractor()
matcher = WorldCoinsMatcher()

class IrisVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Recognition System")
        self.root.geometry("1200x900")
        
        # Initialize the iris recognition pipeline
        self.pipeline = IrisRecognitionPipeline(segmentator, normalizer, feature_extractor, matcher)
        
        # Initialize the database
        self.db = IrisDatabase(self.pipeline, storage_path="Data/iris_db")
        assert self.db is not None, "Failed to initialize the database. Please enroll subjects first."

        
        # Create the main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for Single Mode, Compare Mode, and Verify Mode
        self.single_frame = ttk.Frame(self.notebook)
        self.compare_frame = ttk.Frame(self.notebook)
        self.verify_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.single_frame, text="Single Mode")
        self.notebook.add(self.compare_frame, text="Compare Mode")
        self.notebook.add(self.verify_frame, text="Verify Mode")
        
        # Initialize images and result dictionaries
        self.single_image = None
        self.left_image = None
        self.right_image = None
        self.verify_image = None
        self.single_results = {}
        self.left_results = {}
        self.right_results = {}
        self.verify_results = {}
        
        # Setup the UI for all modes
        self.setup_single_mode()
        self.setup_compare_mode()
        self.setup_verify_mode()
        
        # Bind window resize event with debouncing
        self.resize_timer = None
        self.root.bind("<Configure>", self.on_window_resize)
    
    def setup_single_mode(self):
        # Top frame for buttons and options
        top_frame = ttk.Frame(self.single_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        load_btn = ttk.Button(top_frame, text="Load Image", command=self.load_single_image)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        analyze_btn = ttk.Button(top_frame, text="Analyze", command=self.analyze_single_image)
        analyze_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Display options
        options_frame = ttk.LabelFrame(self.single_frame, text="Display Options", padding="5")
        options_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.single_display_var = tk.StringVar(value="Segmentation")
        display_combobox = ttk.Combobox(options_frame, textvariable=self.single_display_var, values=["Segmentation", "Normalization", "Iris Code"], state="readonly", width=15)
        display_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        display_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_single_display())
        
        self.single_seg_var = tk.StringVar(value="")
        self.single_seg_combobox = ttk.Combobox(options_frame, textvariable=self.single_seg_var, values=[], state="readonly", width=15)
        self.single_seg_combobox.pack_forget()  # Initially hidden
        self.single_seg_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_single_display())
        
        # Display frame with dynamic layout
        self.single_display_frame = ttk.Frame(self.single_frame)
        self.single_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image frame (always displayed)
        self.orig_frame = ttk.LabelFrame(self.single_display_frame, text="Original Image")
        self.orig_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        self.orig_canvas = tk.Canvas(self.orig_frame, bg="white", highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        self.orig_canvas.config(width=300, height=300)  # Minimum size to ensure visibility
        
        # Other frames (shown dynamically)
        self.result_frame = ttk.Frame(self.single_display_frame)  # Frame to hold the dynamic result
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.seg_frame = ttk.LabelFrame(self.result_frame, text="Segmentation")
        self.seg_canvas = tk.Canvas(self.seg_frame, bg="white", highlightthickness=0)
        self.seg_canvas.pack(fill=tk.BOTH, expand=True)
        self.seg_canvas.config(width=300, height=300)  # Minimum size
        
        self.norm_frame = ttk.LabelFrame(self.result_frame, text="Normalization")
        self.norm_canvas = tk.Canvas(self.norm_frame, bg="white", highlightthickness=0)
        self.norm_canvas.pack(fill=tk.BOTH, expand=True)
        self.norm_canvas.config(width=300, height=300)  # Minimum size
        
        self.code_frame = ttk.LabelFrame(self.result_frame, text="Iris Code")
        self.code_canvas = tk.Canvas(self.code_frame, bg="white", highlightthickness=0)
        self.code_canvas.pack(fill=tk.BOTH, expand=True)
        self.code_canvas.config(width=300, height=600)  # Larger height for iris code
    
    def setup_compare_mode(self):
        # Top frame for buttons
        top_frame = ttk.Frame(self.compare_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(top_frame, text="Load Left Image", command=lambda: self.load_compare_image("left")).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(top_frame, text="Load Right Image", command=lambda: self.load_compare_image("right")).pack(side=tk.LEFT, padx=(20, 10))
        
        ttk.Button(top_frame, text="Compare", command=self.compare_images).pack(side=tk.LEFT, padx=(10, 0))
        
        # Display options
        options_frame = ttk.LabelFrame(self.compare_frame, text="Display Options", padding="5")
        options_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.display_var = tk.StringVar(value="Segmentation")  # Default to Segmentation
        display_combobox = ttk.Combobox(options_frame, textvariable=self.display_var, values=["Segmentation", "Normalization", "Iris Code"], state="readonly", width=15)
        display_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        display_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_compare_display())
        
        self.seg_var = tk.StringVar(value="")
        self.seg_combobox = ttk.Combobox(options_frame, textvariable=self.seg_var, values=[], state="readonly", width=15)
        self.seg_combobox.pack_forget()  # Initially hidden
        self.seg_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_compare_display())
        
        # Results frame
        results_frame = ttk.LabelFrame(self.compare_frame, text="Results")
        results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(results_frame, text="Similarity Score:").pack(side=tk.LEFT, padx=5, pady=5)
        self.score_var = tk.StringVar(value="N/A")
        ttk.Label(results_frame, textvariable=self.score_var).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(results_frame, text="Match Status:").pack(side=tk.LEFT, padx=5, pady=5)
        self.match_var = tk.StringVar(value="N/A")
        ttk.Label(results_frame, textvariable=self.match_var).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Main display frame
        self.compare_display_frame = ttk.Frame(self.compare_frame)
        self.compare_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original images frame (always displayed)
        self.orig_frame_compare = ttk.LabelFrame(self.compare_display_frame, text="Original Images")
        self.orig_frame_compare.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        self.left_orig_canvas = tk.Canvas(self.orig_frame_compare, bg="white", highlightthickness=0)
        self.left_orig_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_orig_canvas = tk.Canvas(self.orig_frame_compare, bg="white", highlightthickness=0)
        self.right_orig_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame for the dynamic result display
        self.result_frame_compare = ttk.Frame(self.compare_display_frame)
        self.result_frame_compare.pack(fill=tk.BOTH, expand=True)
        
        self.seg_frame_compare = ttk.LabelFrame(self.result_frame_compare, text="Segmentation Images")
        self.left_seg_canvas = tk.Canvas(self.seg_frame_compare, bg="white", highlightthickness=0)
        self.left_seg_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_seg_canvas = tk.Canvas(self.seg_frame_compare, bg="white", highlightthickness=0)
        self.right_seg_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.norm_frame_compare = ttk.LabelFrame(self.result_frame_compare, text="Normalization Images")
        self.left_norm_canvas = tk.Canvas(self.norm_frame_compare, bg="white", highlightthickness=0)
        self.left_norm_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_norm_canvas = tk.Canvas(self.norm_frame_compare, bg="white", highlightthickness=0)
        self.right_norm_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.code_frame_compare = ttk.LabelFrame(self.result_frame_compare, text="Iris Code Images")
        self.left_code_canvas = tk.Canvas(self.code_frame_compare, bg="white", highlightthickness=0)
        self.left_code_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_code_canvas = tk.Canvas(self.code_frame_compare, bg="white", highlightthickness=0)
        self.right_code_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_verify_mode(self):
        # Top frame for buttons and options
        top_frame = ttk.Frame(self.verify_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        load_btn = ttk.Button(top_frame, text="Load Image", command=self.load_verify_image)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        verify_btn = ttk.Button(top_frame, text="Verify", command=self.verifying_image)
        verify_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Display options
        options_frame = ttk.LabelFrame(self.verify_frame, text="Display Options", padding="5")
        options_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.verify_display_var = tk.StringVar(value="Segmentation")
        display_combobox = ttk.Combobox(options_frame, textvariable=self.verify_display_var, values=["Segmentation", "Normalization", "Iris Code"], state="readonly", width=15)
        display_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        display_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_verify_display())
        
        self.verify_seg_var = tk.StringVar(value="")
        self.verify_seg_combobox = ttk.Combobox(options_frame, textvariable=self.verify_seg_var, values=[], state="readonly", width=15)
        self.verify_seg_combobox.pack_forget()  # Initially hidden
        self.verify_seg_combobox.bind("<<ComboboxSelected>>", lambda _: self.update_verify_display())
        
        # Results frame
        results_frame = ttk.LabelFrame(self.verify_frame, text="Verification Results")
        results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(results_frame, text="Identity:").pack(side=tk.LEFT, padx=5, pady=5)
        self.identity_var = tk.StringVar(value="N/A")
        ttk.Label(results_frame, textvariable=self.identity_var).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(results_frame, text="Score:").pack(side=tk.LEFT, padx=5, pady=5)
        self.verify_score_var = tk.StringVar(value="N/A")
        ttk.Label(results_frame, textvariable=self.verify_score_var).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Display frame with dynamic layout
        self.verify_display_frame = ttk.Frame(self.verify_frame)
        self.verify_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image frame (always displayed)
        self.verify_orig_frame = ttk.LabelFrame(self.verify_display_frame, text="Original Image")
        self.verify_orig_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        self.verify_orig_canvas = tk.Canvas(self.verify_orig_frame, bg="white", highlightthickness=0)
        self.verify_orig_canvas.pack(fill=tk.BOTH, expand=True)
        self.verify_orig_canvas.config(width=300, height=300)  # Minimum size
        
        # Other frames (shown dynamically)
        self.verify_result_frame = ttk.Frame(self.verify_display_frame)
        self.verify_result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.verify_seg_frame = ttk.LabelFrame(self.verify_result_frame, text="Segmentation")
        self.verify_seg_canvas = tk.Canvas(self.verify_seg_frame, bg="white", highlightthickness=0)
        self.verify_seg_canvas.pack(fill=tk.BOTH, expand=True)
        self.verify_seg_canvas.config(width=300, height=300)  # Minimum size
        
        self.verify_norm_frame = ttk.LabelFrame(self.verify_result_frame, text="Normalization")
        self.verify_norm_canvas = tk.Canvas(self.verify_norm_frame, bg="white", highlightthickness=0)
        self.verify_norm_canvas.pack(fill=tk.BOTH, expand=True)
        self.verify_norm_canvas.config(width=300, height=300)  # Minimum size
        
        self.verify_code_frame = ttk.LabelFrame(self.verify_result_frame, text="Iris Code")
        self.verify_code_canvas = tk.Canvas(self.verify_code_frame, bg="white", highlightthickness=0)
        self.verify_code_canvas.pack(fill=tk.BOTH, expand=True)
        self.verify_code_canvas.config(width=300, height=600)  # Larger height for iris code
    
    def process_image(self, filepath):
        self.pipeline.set_image(filepath)
        seg_images = self.pipeline.get_segmentation_images()
        norm_path = self.pipeline.get_normalization_image()
        code_path = self.pipeline.get_iris_code_image()
        return {
            'seg_images': seg_images,
            'norm_path': norm_path,
            'code_path': code_path
        }
    
    def load_single_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if filepath:
            try:
                img = cv2.imread(filepath)
                if img is None:
                    messagebox.showerror("Error", "Failed to load the image.")
                    return
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.single_image = img
                self.single_results = {}
                self.update_single_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the image: {str(e)}")
    
    def load_compare_image(self, side):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if filepath:
            try:
                img = cv2.imread(filepath)
                if img is None:
                    messagebox.showerror("Error", "Failed to load the image.")
                    return
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if side == "left":
                    self.left_image = img
                    self.left_results = {}
                else:
                    self.right_image = img
                    self.right_results = {}
                self.update_compare_display()
                self.score_var.set("N/A")
                self.match_var.set("N/A")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the image: {str(e)}")
    
    def load_verify_image(self):
        print("Loading image for verification...")
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if filepath:
            try:
                img = cv2.imread(filepath)  # BGR
                if img is None:
                    messagebox.showerror("Error", "Failed to load the image.")
                    return
                self.verify_image = img  # Store as BGR
                self.verify_results = {}
                self.identity_var.set("N/A")
                self.verify_score_var.set("N/A")
                self.update_verify_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the image: {str(e)}")
        print("Image loaded successfully.")
    
    def analyze_single_image(self):
        if self.single_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
            filepath = "temp_single_image.png"
            cv2.imwrite(filepath, cv2.cvtColor(self.single_image, cv2.COLOR_RGB2BGR))
            self.single_results = self.process_image(filepath)
            self.single_seg_combobox["values"] = [name for name, _ in self.single_results['seg_images']] if self.single_results['seg_images'] else []
            if self.single_results['seg_images']:
                self.single_seg_var.set(self.single_results['seg_images'][0][0])
            self.update_single_display()
            os.remove(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def verifying_image(self):
        print("Verifying image...")
        if self.verify_image is None:
            print("No image loaded.")
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        print("Image is loaded.")
        if self.db.is_database_empty():
            print("Database is empty.")
            messagebox.showwarning("Warning", "Database is empty. Please enroll subjects first.")
            return
        print("Database has entries.")
        try:
            filepath = "temp_verify_image.png"
            print("Saving image to:", filepath)
            cv2.imwrite(filepath, self.verify_image)
            self.verify_results = self.process_image(filepath)
            self.verify_seg_combobox["values"] = [name for name, _ in self.verify_results['seg_images']] if self.verify_results['seg_images'] else []
            if self.verify_results['seg_images']:
                self.verify_seg_var.set(self.verify_results['seg_images'][0][0])
            
            print("Identifying image...")
            subject_id, name, score, eye = self.db.identify(filepath)
            print(f"Identified subject ID: {subject_id}, Name: {name}, Score: {score}, Eye: {eye}")
            if subject_id:
                self.identity_var.set(f"{name} ({eye} eye)")
                self.verify_score_var.set(f"{score:.4f}")
            else:
                self.identity_var.set("Not Recognized")
                self.verify_score_var.set(f"{score:.4f}")
            
            self.update_verify_display()
            os.remove(filepath)
            print("Verification complete.")
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            messagebox.showerror("Error", f"Verification failed: {str(e)}")
    
    def update_single_display(self):
        if self.single_image is None:
            return
        
        # Always display original image
        self.display_image(self.orig_canvas, self.single_image)
        
        # Hide all other frames initially
        self.seg_frame.pack_forget()
        self.norm_frame.pack_forget()
        self.code_frame.pack_forget()
        self.single_seg_combobox.pack_forget()  # Hide segmentation dropdown
        
        # Show selected frame and segmentation dropdown if applicable
        display_choice = self.single_display_var.get()
        if display_choice == "Segmentation" and 'seg_images' in self.single_results:
            selected_seg = self.single_seg_var.get()
            seg_img_path = next((path for name, path in self.single_results['seg_images'] if name == selected_seg), None)
            if seg_img_path:
                seg_img = cv2.imread(seg_img_path)
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                self.display_image(self.seg_canvas, seg_img)
            self.seg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.single_seg_combobox.pack(side=tk.LEFT, padx=5, pady=5)  # Show segmentation dropdown
        elif display_choice == "Normalization" and 'norm_path' in self.single_results:
            norm_img = cv2.imread(self.single_results['norm_path'])
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.norm_canvas, norm_img)
            self.norm_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        elif display_choice == "Iris Code" and 'code_path' in self.single_results:
            code_img = cv2.imread(self.single_results['code_path'])
            code_img = cv2.cvtColor(code_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.code_canvas, code_img, is_iris_code=True)
            self.code_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    def compare_images(self):
        if self.left_image is None or self.right_image is None:
            messagebox.showwarning("Warning", "Please load both images first.")
            return
        try:
            left_filepath = "temp_left_image.png"
            right_filepath = "temp_right_image.png"
            cv2.imwrite(left_filepath, cv2.cvtColor(self.left_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(right_filepath, cv2.cvtColor(self.right_image, cv2.COLOR_RGB2BGR))
            
            self.left_results = self.process_image(left_filepath)
            self.right_results = self.process_image(right_filepath)
            self.seg_combobox["values"] = [name for name, _ in self.left_results['seg_images']] if self.left_results['seg_images'] else []
            if self.left_results['seg_images']:
                self.seg_var.set(self.left_results['seg_images'][0][0])
            
            score = self.pipeline.getScore(left_filepath, right_filepath)
            self.update_compare_display()
            self.score_var.set(f"{score:.4f}")
            self.match_var.set("MATCH" if score < self.pipeline.get_threshold() else "NO MATCH")
            os.remove(left_filepath)
            os.remove(right_filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
    
    def update_compare_display(self):
        if self.left_image is not None or self.right_image is not None:
            # Always display original images
            self.display_image(self.left_orig_canvas, self.left_image if self.left_image is not None else None)
            self.display_image(self.right_orig_canvas, self.right_image if self.right_image is not None else None)
            
            # Hide all result frames initially
            self.seg_frame_compare.pack_forget()
            self.norm_frame_compare.pack_forget()
            self.code_frame_compare.pack_forget()
            self.seg_combobox.pack_forget()  # Hide segmentation dropdown
            
            # Show only the selected frame
            display_choice = self.display_var.get()
            if display_choice == "Segmentation" and ('seg_images' in self.left_results or 'seg_images' in self.right_results):
                selected_seg = self.seg_var.get()
                if self.left_image is not None and 'seg_images' in self.left_results:
                    seg_img_path = next((path for name, path in self.left_results['seg_images'] if name == selected_seg), None)
                    if seg_img_path:
                        seg_img = cv2.imread(seg_img_path)
                        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                        self.display_image(self.left_seg_canvas, seg_img)
                if self.right_image is not None and 'seg_images' in self.right_results:
                    seg_img_path = next((path for name, path in self.right_results['seg_images'] if name == selected_seg), None)
                    if seg_img_path:
                        seg_img = cv2.imread(seg_img_path)
                        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                        self.display_image(self.right_seg_canvas, seg_img)
                self.seg_frame_compare.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
                self.seg_combobox.pack(side=tk.LEFT, padx=5, pady=5)  # Show segmentation dropdown
            elif display_choice == "Normalization" and ('norm_path' in self.left_results or 'norm_path' in self.right_results):
                if self.left_image is not None and 'norm_path' in self.left_results:
                    norm_img = cv2.imread(self.left_results['norm_path'])
                    norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                    self.display_image(self.left_norm_canvas, norm_img)
                if self.right_image is not None and 'norm_path' in self.right_results:
                    norm_img = cv2.imread(self.right_results['norm_path'])
                    norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                    self.display_image(self.right_norm_canvas, norm_img)
                self.norm_frame_compare.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            elif display_choice == "Iris Code" and ('code_path' in self.left_results or 'code_path' in self.right_results):
                if self.left_image is not None and 'code_path' in self.left_results:
                    code_img = cv2.imread(self.left_results['code_path'])
                    code_img = cv2.cvtColor(code_img, cv2.COLOR_BGR2RGB)
                    self.display_image(self.left_code_canvas, code_img, is_iris_code=True)
                if self.right_image is not None and 'code_path' in self.right_results:
                    code_img = cv2.imread(self.right_results['code_path'])
                    code_img = cv2.cvtColor(code_img, cv2.COLOR_BGR2RGB)
                    self.display_image(self.right_code_canvas, code_img, is_iris_code=True)
                self.code_frame_compare.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    def update_verify_display(self):
        if self.verify_image is None:
            return
        
        # Always display original image
        self.display_image(self.verify_orig_canvas, cv2.cvtColor(self.verify_image, cv2.COLOR_BGR2RGB))
        
        # Hide all other frames initially
        self.verify_seg_frame.pack_forget()
        self.verify_norm_frame.pack_forget()
        self.verify_code_frame.pack_forget()
        self.verify_seg_combobox.pack_forget()  # Hide segmentation dropdown
        
        # Show selected frame and segmentation dropdown if applicable
        display_choice = self.verify_display_var.get()
        if display_choice == "Segmentation" and 'seg_images' in self.verify_results:
            selected_seg = self.verify_seg_var.get()
            seg_img_path = next((path for name, path in self.verify_results['seg_images'] if name == selected_seg), None)
            if seg_img_path:
                seg_img = cv2.imread(seg_img_path)
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                self.display_image(self.verify_seg_canvas, seg_img)
            self.verify_seg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.verify_seg_combobox.pack(side=tk.LEFT, padx=5, pady=5)  # Show segmentation dropdown
        elif display_choice == "Normalization" and 'norm_path' in self.verify_results:
            norm_img = cv2.imread(self.verify_results['norm_path'])
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.verify_norm_canvas, norm_img)
            self.verify_norm_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        elif display_choice == "Iris Code" and 'code_path' in self.verify_results:
            code_img = cv2.imread(self.verify_results['code_path'])
            code_img = cv2.cvtColor(code_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.verify_code_canvas, code_img, is_iris_code=True)
            self.verify_code_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    def display_image(self, canvas, image, is_iris_code=False):
        if image is None:
            canvas.delete("all")
            return
        
        # Get current canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ensure canvas has a reasonable size if not yet rendered
        if canvas_width <= 1:
            canvas_width = 400  # Default width
        if canvas_height <= 1:
            canvas_height = 400 if not is_iris_code else 800  # Default height
        
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        img_width, img_height = pil_img.size
        
        # Calculate aspect ratio to fit the image within the canvas
        aspect_ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * aspect_ratio)
        new_height = int(img_height * aspect_ratio)
        
        # Resize image while preserving aspect ratio
        resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_img)
        
        # Center the image in the canvas
        x = canvas_width // 2
        y = canvas_height // 2
        
        canvas.delete("all")
        canvas.create_image(x, y, anchor=tk.CENTER, image=photo)
        canvas.image = photo  # Keep reference to avoid garbage collection
    
    def clear_canvas(self, canvas):
        canvas.delete("all")
    
    def on_window_resize(self, event):
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(200, self.update_all_displays)
    
    def update_all_displays(self):
        if self.single_image is not None:
            self.update_single_display()
        if self.left_image is not None or self.right_image is not None:
            self.update_compare_display()
        if self.verify_image is not None:
            self.update_verify_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = IrisVisualizer(root)
    root.mainloop()