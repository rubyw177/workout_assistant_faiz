# Advanced Pose Estimation and CSV Data Generation

## Overview
This project implements a **pose estimation pipeline** using **TensorFlow MoveNet**, extracting **body keypoints** from images and storing the results in a structured CSV format. The pipeline is designed to work with **human movement classification**, specifically for **workout exercise tracking**. 

## Features
- **Pose Estimation**: Extracts **17 keypoints** from human images.
- **Deep Learning**: Utilizes TensorFlow Hub's **MoveNet SinglePose Lightning** model.
- **Data Processing**: Converts pose estimation results into a **structured dataset**.
- **Visualization**: Overlays keypoints on images for validation.
- **Dataset Generation**: Automates **CSV file creation** for further analysis.

## Project Structure

```
📂 project_root/
├── 📂 dataset/               # Directory containing categorized images
│   ├── 📂 push_up/           # Images labeled as 'push up'
│   ├── 📂 squat/             # Images labeled as 'squat'
│   ├── ...                   # Other exercise classes (if applicable)
├── 📝 README.md              # Project documentation
├── 📜 csv_generator.py       # Main script for keypoint extraction and CSV generation
├── 📓 body_keypoints_calculator.ipynb  # Notebook for visualizing keypoint extraction
├── 📓 pose_model.ipynb       # Model development and evaluation notebook
└── 📂 output/                # Output files including CSV datasets
```

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install tensorflow numpy pandas opencv-python matplotlib imageio tensorflow-hub
```

## Usage
### 1. Extract Keypoints from Images
Run the `csv_generator.py` script to extract pose keypoints and save them into a structured CSV file.
```bash
python csv_generator.py
```
- The script scans subdirectories inside the **dataset** folder, processes images, and generates labeled **keypoints CSV data**.
- **Keypoints Mapping:**
  - Nose, Eyes, Ears, Shoulders, Elbows, Wrists
  - Hips, Knees, Ankles
- **Example Output CSV Format:**

| noseX | noseY | left_shoulderX | left_shoulderY | right_shoulderX | right_shoulderY | left_hipX | left_hipY | right_hipX | right_hipY | exercise |
|-------|-------|----------------|----------------|-----------------|-----------------|-----------|-----------|-----------|-----------|----------|
| 0.45  | 0.62  | 0.38           | 0.52           | 0.55            | 0.51            | 0.40      | 0.74      | 0.58      | 0.73      | squat    |
| 0.46  | 0.61  | 0.37           | 0.50           | 0.54            | 0.49            | 0.41      | 0.75      | 0.57      | 0.74      | push up  |

### 2. Visualize Keypoints
To visualize the extracted keypoints on images, run:
```python
python -c 'from csv_generator import display_keypoints'
```
This function overlays detected keypoints and skeletal connections on the image.

### 3. Model Development and Evaluation
- Use `pose_model.ipynb` to process the extracted keypoints and analyze movement patterns.
- Apply keypoints for **further applications**, such as movement tracking or anomaly detection.

## Implementation Details
### MoveNet Model
- The **MoveNet SinglePose Lightning** model is loaded via TensorFlow Hub:
  ```python
  model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
  ```
- The input image is resized to `192x192` for inference.
- The model outputs **17 keypoints**, each represented as `(y, x, confidence score)`.

### Data Processing Pipeline
1. **Read Image** → Convert to TensorFlow format.
2. **Resize & Normalize** → Prepare for inference.
3. **Predict Keypoints** → Run inference with MoveNet.
4. **Extract & Structure Data** → Convert keypoints into CSV format.
5. **Save & Label Data** → Assign exercise labels based on folder structure.
