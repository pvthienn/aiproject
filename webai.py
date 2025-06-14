import gradio as gr
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO('./weights/best.pt')

# Load DeepSORT tracker

trackeri = DeepSort(max_age=30)
# Global variables
streaming_flag = False
video_path = ""
counted_ids = set()
BAGGED_PEAR_CLASS_ID = 0  # Class ID for "Bagged Pears"

def process_file(file=None):
    """Process uploaded file, extract metadata, and save an Excel file."""
    file_path = file.name
    file_type = "Video" if file_path.endswith(('.mp4', '.avi', '.mov')) else "Image"
    
    metadata = {
        "File Name": [os.path.basename(file_path)],
        "File Type": [file_type],
        "File Size (KB)": [round(os.path.getsize(file_path) / 1024, 2)]
    }
    
    # Save metadata to Excel
    excel_path = "output_metadata.xlsx"
    df = pd.DataFrame(metadata)
    df.to_excel(excel_path, index=False)
    
    return excel_path, file_path if file_type == "Video" else None

def stream_camera():

    global streaming_flag, video_path
    streaming_flag = True
    """Capture and return video frames from the camera with real-time object detection and tracking."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frames = []  #
    counted_ids.clear() 
    tracker = DeepSort(max_age=30)
    while streaming_flag:
        ret, frame = cap.read()
        if not ret:
            break
        frame= cv2.resize(frame, (640, 640))
        results = model.predict(frame, conf=0.45, imgsz=640)[0]
        detections = []

        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls = track.det_class  # Get the class ID from DeepSORT

            # Count Bagged Pears
            if cls == BAGGED_PEAR_CLASS_ID:
                counted_ids.add(track_id)

                # Draw box and label
                label = f"ID {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw total count of Bagged Pears
        count_text = f"Bagged Pears Count: {len(counted_ids)}"
        cv2.putText(frame, count_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # Convert frame to RGB for Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(np.array(pil_image))
        yield pil_image

    cap.release()

    # Save the frames as a video after streaming finishes
    if frames:
        video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_path = video_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 640))  # Adjust FPS and resolution if needed

        # Convert PIL images to OpenCV format and write them to the video
        for frame in frames:
            frame_cv = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
            out.write(frame_bgr)

        out.release()
    return video_path  # Return the path to the saved video file
def stream_video(file):
    """Process the uploaded video file for object detection and tracking."""
    cap = cv2.VideoCapture(file.name)  # Open the uploaded video file
    frames = []
    counted_ids.clear()  # Reset counted ids
    tracker = DeepSort(max_age=30)
    
    id_map = {}          # Ánh xạ track_id gốc -> display_id dễ hiểu
    next_display_id = 1  # ID hiển thị tuần tự
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        results = model.predict(frame, conf=0.45, imgsz=640)[0]
        detections = []

        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Gán display_id nếu chưa có
            if track_id not in id_map:
                id_map[track_id] = next_display_id
                next_display_id += 1

            display_id = id_map[track_id]

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls = track.det_class  # Lấy class ID từ DeepSORT

            # Chỉ vẽ nếu là đối tượng cần đếm (ví dụ: BAGGED_PEAR)
            if cls == BAGGED_PEAR_CLASS_ID:
                counted_ids.add(track_id)

                # Tìm confidence score gần đúng
                conf_score = None
                track_box = [x1, y1, x2 - x1, y2 - y1]
                for det_box, det_conf, det_cls in detections:
                    if cls == det_cls and np.allclose(det_box, track_box, atol=5):
                        conf_score = det_conf
                        break

                if conf_score is not None:
                    label = f"ID {display_id} ({conf_score:.2f})"
                else:
                    label = f"ID {display_id}"

                # Vẽ bounding box và label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hiển thị tổng số Bagged Pears
        count_text = f"Bagged Pears Count: {len(counted_ids)}"
        cv2.putText(frame, count_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Convert frame sang RGB cho Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(np.array(pil_image))

    cap.release()

    # Save the frames as a video after processing
    video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_path = video_file.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 640))  # Adjust FPS and resolution if needed

    # Convert PIL images to OpenCV format and write them to the video
    for frame in frames:
        frame_cv = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
        out.write(frame_bgr)

    out.release()

    # Return the path of the saved video
    return video_path
def process_image(file):
    """Process the uploaded image file for object detection and tracking."""
    counted_ids.clear()
    
    file_path = file.name  # Get the file path
    image_np = cv2.imread(file_path)  # Open the image file
    image_np= cv2.resize(image_np, (640, 640))
    # Convert the image to a numpy array (required by the YOLO model)
    image_np = np.array(image_np)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Perform object detection using YOLO model
    results = model.predict(image_np,  conf=0.25)[0]
    detections = []
    
    # Prepare the detections for DeepSORT tracking
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        cls = int(cls)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls, None))

    # Update the tracker with detections
    tracks = trackeri.update_tracks(detections, frame=image_np)

    # Draw bounding boxes and track labels
      # Reset counted ids for new image processing
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cls = track.det_class  # Get the class ID from DeepSORT

        # Count Bagged Pears
        if cls == BAGGED_PEAR_CLASS_ID:
            counted_ids.add(track_id)

            # Draw box and label
            label = f"ID {track_id}"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_np, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Draw total count of Bagged Pears on the image
    count_text = f"Bagged Pears Count: {len(counted_ids)}"
    cv2.putText(image_np, count_text, (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Convert processed image back to PIL format for Gradio output
    #processed_image = Image.fromarray(image_np)
    processed_image = np.array(image_np)
    #frame_bgr = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
    # Generate metadata for the image (file size, bagged pear count)
    
    return processed_image# Return the processed image and Excel metadata file
def finish_streaming():
    global streaming_flag
    streaming_flag = False  # Set the flag to False to stop streaming
    return video_path

# Custom CSS for a small button
small_button_css = """
<style>
#start_btn button {
    padding: 2px 8px !important;
    font-size: 12px !important;
    width: 60px !important;
}
</style>
"""

with gr.Blocks(css="""
    .gradio-container {background-color: #FFD580 !important;}
    .markdown-center {color: white !important; font-weight: bold !important;}
""") as demo:
    gr.Markdown("## Project of Counting Pears", elem_classes=["markdown-center"])
    
    with gr.Tabs():
        with gr.TabItem("Streaming"):
            with gr.Column():
                with gr.Row(): 
                    start_button = gr.Button("▶ Start", elem_id="start_btn", min_width=80)
                with gr.Row(): 
                    with gr.Column(scale=4):
                        camera_input = gr.Image(label="Streaming Video", min_width=120)
                    with gr.Column(scale=1):
                        file_button = gr.Button("📂 Get Output File")
                        excel_output1 = gr.File(label="Get Metadata Excel")
                with gr.Row(): 
                    finish_button = gr.Button("⏹ Stop")  # Button to finish and save the video
                
            video_output = gr.Image(label="📂 Saved Video")  #gr.Video(label="Processed Video")
            start_button.click(fn=stream_camera, inputs=None, outputs=[camera_input])
            finish_button.click(fn=finish_streaming, inputs=None, outputs=[video_output])

            file_button.click(fn=process_file, inputs=None, outputs=excel_output1)

        with gr.TabItem("Already_Image/Video"):
            with gr.Row():
                with gr.Column(scale=5):
                    file_input = gr.File(label="Upload Image or Video")  # User uploads an image or video
                with gr.Column(scale=1):
                    excel_output = gr.File(label="Get Metadata Excel")  # Output metadata (Excel file)
            
            with gr.Row():
                video_output = gr.Video(label="Output Image or Video (if applicable)")  # To display the processed video or image
                #video_output = gr.Image(label="Output Image or Video (if applicable)")
            
                process_button = gr.Button("Submit")  # Process the file upon clicking the button
            #process_button.click(fn=stream_camera, inputs=None, outputs=[camera_input])
            # Adjust the processing function to handle both image and video uploads
            def process_uploaded_file(file=None):
                file_path = file.name
                file_type = "Video" if file_path.endswith(('.mp4', '.avi', '.mov')) else "Image"
                
                # For video processing (YOLO + DeepSORT tracking or similar processing)
                if file_type == "Video":
                    # You can modify the following function to handle video processing with YOLO + DeepSORT
                    video_output_path = stream_video(file)
                    metadata = {
                        "File Name": [os.path.basename(file_path)],
                        "Numer of Pear_Bags": [len(counted_ids)],
                        "File Size (KB)": [round(os.path.getsize(file_path) / 1024, 2)]
                    }
                    # Generate metadata Excel
                    excel_path = "output_metadata.xlsx"
                    df = pd.DataFrame(metadata)
                    df.to_excel(excel_path, index=False)
                    counted_ids.clear() 
                    return video_output_path, excel_path
                
                # For image processing (basic display or detection)
                elif file_type == "Image":
                    video_output_path = process_image(file)
                    metadata = {
                        "File Name": [os.path.basename(file_path)],
                        "Number of Pear Bags": [len(counted_ids)],
                        "File Size (KB)": [round(os.path.getsize(file_path) / 1024, 2)]
                    }

                    # Save metadata to Excel
                    excel_path = "output_metadata.xlsx"
                    df = pd.DataFrame(metadata)
                    df.to_excel(excel_path, index=False)
                    #counted_ids.clear() 
                    return video_output_path, excel_path

            # Connect the process button to the function
            process_button.click(fn=process_uploaded_file, inputs=file_input, outputs=[video_output, excel_output])

    gr.Markdown("### Instructions:", elem_classes=["markdown-center"])
    gr.Markdown("1. Upload an image or video file.", elem_classes=["markdown-center"])
    gr.Markdown("2. Click 'Submit' to detect and track objects.", elem_classes=["markdown-center"])
    gr.Markdown("3. If a video or image is uploaded, it will be displayed below.", elem_classes=["markdown-center"])
    gr.Markdown("4. You can download the metadata file in Excel format after processing.", elem_classes=["markdown-center"])

demo.launch()
