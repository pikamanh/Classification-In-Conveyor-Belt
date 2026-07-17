# Classification on a Conveyor Belt

An AI-assisted conveyor-belt sorting system that detects and tracks geometric objects from an IP camera stream, counts them, and instructs an Arduino-controlled servo to push each object in the appropriate direction. The current application uses an Ultralytics YOLO checkpoint to distinguish rectangular and triangular objects and ByteTrack to prevent the same object from being processed more than once.

> The current setup is calibrated for a fixed camera position, a specific conveyor layout, and two object classes. Camera credentials, crop coordinates, timing, and the model path must be reviewed before running the system on different hardware.

![Conveyor-belt classification demo](VideoForReadme/Demo.gif)

## Features

- Reads an RTSP stream in a background thread with OpenCV and FFmpeg.
- Automatically uses CUDA when PyTorch detects a compatible GPU, otherwise falls back to the CPU.
- Runs object detection with a bundled Ultralytics YOLO checkpoint.
- Tracks detections across frames with Supervision's ByteTrack implementation.
- Uses a virtual trigger line and a configurable delay to synchronize detections with the downstream servo.
- Sends serial commands to an Arduino at `9600` baud.
- Sorts two object categories:
  - Rectangle: serial command `1`, servo turns from 90° toward 180°.
  - Triangle: serial command `2`, servo turns from 90° toward 0°.
- Displays live rectangle and triangle counters in an OpenCV window.
- Includes an auxiliary script for exporting one camera frame per second.
- Includes several trained model checkpoints and the Arduino servo sketch.

## Processing Pipeline

```text
IP camera (RTSP)
        │
        ▼
Read frames in a background thread
        │
        ▼
Crop the configured conveyor region
        │
        ▼
Detect objects with YOLO
        │
        ▼
Assign persistent IDs with ByteTrack
        │
        ▼
Detect when an object reaches the trigger line
        │
        ▼
Wait for the object to travel to the sorting point
        │
        ▼
Send command 1 or 2 to the Arduino over USB serial
        │
        ▼
Move the servo, update the counter, and mark the object as processed
```

The video reader keeps up to five frames in memory. In the default configuration, YOLO processes a crop around `x=500..800` and `y=480..870`, derived from the detection coordinates in `main.py`. A detection is accepted when its confidence is at least `0.8`. When the top edge of a new tracked object reaches `y=60` inside the cropped image, the application starts a `5.5`-second timer before sending the sorting command.

## Technology Stack

- **Python** and **PyTorch** for the application and inference runtime.
- **Ultralytics YOLO** for object detection.
- **Supervision ByteTrack** for multi-object tracking.
- **OpenCV** and **NumPy** for RTSP capture, cropping, display, and image processing.
- **PySerial** for USB serial communication with the Arduino.
- **Arduino Servo library** for actuator control.

## Hardware Requirements

- A computer capable of running the supplied YOLO model.
- An NVIDIA GPU is recommended for better throughput, but is not required by the current device-selection logic.
- An RTSP-capable IP camera. The original setup uses a TP-Link Tapo camera.
- The computer and camera must be reachable on the same network.
- A conveyor belt positioned consistently relative to the camera.
- An Arduino-compatible board connected to the computer over USB.
- A servo connected to Arduino digital pin `9`.

The hard-coded crop reaches approximately `x=930` and `y=870` in the original frame, so the input stream must be large enough for those coordinates. Recalibrate the crop when using another camera resolution or mounting position.

## Software Requirements

- Python with support for PyTorch 2.2.
- OpenCV built with FFmpeg support for RTSP input.
- Arduino IDE if the servo sketch needs to be uploaded or modified.
- A compatible NVIDIA driver when using the CUDA 11.8 PyTorch build.

The repository was written primarily for Windows. The selected model is referenced as `model\best new v2.pt`; on Linux or macOS, change it to `model/best new v2.pt` in `main.py`.

## Installation

### 1. Clone the repository

Using HTTPS:

```bash
git clone https://github.com/pikamanh/Classification-In-Conveyor-Belt.git
cd Classification-In-Conveyor-Belt
```

Or using SSH:

```bash
git clone git@github.com:pikamanh/Classification-In-Conveyor-Belt.git
cd Classification-In-Conveyor-Belt
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch

The dependency configuration targets PyTorch 2.2.0 with CUDA 11.8:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For a CPU-only machine or a different CUDA version, install the appropriate PyTorch build for that environment instead.

### 4. Install the application packages

```bash
python -m pip install opencv-python numpy ultralytics supervision pyserial
```

The current `requirements.txt` is not directly portable: it lists `cv2` and `serial` by their import names, and also lists the Python standard-library modules `queue`, `threading`, and `time`. The corresponding installable packages are `opencv-python` and `pyserial`; standard-library modules must not be installed separately.

## Camera Configuration

Configure the camera's RTSP service according to its manufacturer documentation. For a Tapo camera, create a camera account in the Tapo application and obtain the device IP address.

Update the stream URL in `main.py`:

```python
cap = VideoCaptureThread(
    "rtsp://username:password@camera-ip-address:554/stream1"
)
```

Test the URL with VLC or FFmpeg if the application cannot open the stream. Avoid networks that isolate wireless clients from one another.

> Never commit a real RTSP username or password. `export_frames.py` currently contains a hard-coded camera URL; replace it with your own local configuration before use. If the committed credentials are real, rotate them.

For a local webcam, replace the threaded RTSP capture with an OpenCV camera source and adapt the read/release calls. The repository contains a commented `cv2.VideoCapture(0)` example, but webcam mode is not wired into the current main loop.

## Arduino and Servo Setup

Open `Sweep/Servo/Servo.ino` in the Arduino IDE, select the correct board and USB port, and upload the sketch. The sketch:

- opens serial communication at `9600` baud;
- attaches the servo to pin `9`;
- initializes the servo at 90°;
- handles command `1` by waiting `delayTime_1`, turning toward 180°, pausing for one second, and returning to 90°;
- handles command `2` by waiting `delayTime_2`, turning toward 0°, pausing for one second, and returning to 90°.

The default Arduino-side delays are:

| Setting | Default | Current purpose |
| --- | ---: | --- |
| `delayTime_1` | 3000 ms | Rectangle sorting delay |
| `delayTime_2` | 1000 ms | Triangle sorting delay |

Adjust these values to match the distance and speed between the camera trigger line and each sorting point, then upload the sketch again.

![Servo wiring diagram](Sweep/images/sweep_schem.png)

The application selects the first serial port returned by PySerial. Disconnect unrelated serial devices or update `get_arduino_port()` to select the Arduino by port name, USB vendor ID, or product ID when more than one serial device is connected.

## Running the Application

Review the RTSP URL, model path, crop coordinates, and timing first, then run from the repository root:

```bash
python main.py
```

During operation:

1. The terminal reports whether inference is running on CUDA or the CPU and whether an Arduino was found.
2. The `Count` window shows the configured conveyor area with the rectangle and triangle totals.
3. The `Total` window shows the detection crop and red trigger line.
4. Press `q` while an OpenCV window is focused to stop the application.

The counters are incremented only when an Arduino connection exists and a serial command is sent successfully.

## Runtime Configuration

The main parameters are currently defined directly in the source code:

| Parameter | Location | Default |
| --- | --- | --- |
| RTSP URL | `VideoCaptureThread(...)` in `main.py` | Placeholder Tapo URL using `/stream1` |
| YOLO checkpoint | `YOLO(...)` in `main.py` | `model\best new v2.pt` |
| Detection crop | `crop_*_detect` in `main.py` | x: `530..780`, y: `500..850` |
| Effective belt crop padding | `belt()` in `main.py` | x: `-30/+20`, y: `-20/+20` |
| Detection confidence | Detection loop in `main.py` | `0.8` |
| Trigger line | `line_y` in `main.py` | `60` pixels |
| Travel delay before serial command | Detection loop in `main.py` | `5.5` seconds |
| Serial baud rate | Python and Arduino code | `9600` |
| ByteTrack activation threshold | `ByteTrack(...)` in `main.py` | `0.3` |
| ByteTrack matching threshold | `ByteTrack(...)` in `main.py` | `0.8` |
| ByteTrack lost-track buffer | `ByteTrack(...)` in `main.py` | `90` frames |
| Display size for counters | `count()` in `main.py` | `1080 × 720` |

The Python class-to-command mapping is based on numeric class IDs: class `0` is treated as a rectangle, while every other class ID is treated as a triangle. Replacement checkpoints must preserve that mapping or the dispatch logic must be updated.

## Exporting Camera Frames

`export_frames.py` is a small dataset-collection utility. It connects to an RTSP camera, displays the stream, and saves one JPEG image per second to a local `frames/` directory.

Before running it, replace the embedded RTSP URL, then execute:

```bash
python export_frames.py
```

Press `q` to stop recording. The generated `frames/` directory is not currently listed in `.gitignore`, so review it before staging repository changes.

## Project Structure

```text
.
├── main.py                         # Detection, tracking, counting, and serial control
├── export_frames.py                # RTSP frame-export utility
├── requirements.txt                # Original dependency list and CUDA package index
├── model/
│   ├── best new v2.pt              # Checkpoint selected by main.py
│   └── *.pt                        # Alternative YOLO checkpoints
├── Sweep/
│   ├── Servo/
│   │   └── Servo.ino               # Arduino servo controller
│   └── images/                     # Servo reference diagrams
├── imageForReadme/                 # Setup screenshots from the original guide
└── VideoForReadme/
    └── Demo.gif                    # Project demonstration
```

The repository contains inference checkpoints but does not include the training dataset, dataset configuration, or a model-training script.

## Models

The application currently loads `model/best new v2.pt`, an Ultralytics object-detection checkpoint. The `model/` directory also contains earlier or alternative checkpoints, including `best new v1.pt`, `best new v3.pt`, `best v4.pt` through `best v7.pt`, several `box` checkpoints, and `test_coco.pt`.

To use another model, change the path passed to `YOLO(...)`. Ensure that the checkpoint is compatible with the installed Ultralytics version and that its class IDs match the sorting logic.

## Current Limitations

- The RTSP URL, model path, crop coordinates, trigger line, and delays are hard-coded.
- The camera must remain in a fixed position and provide a sufficiently large frame for the configured crop.
- Only two sorting outcomes are implemented, and every class other than class `0` is routed as a triangle.
- The application uses the first enumerated serial port instead of identifying the Arduino explicitly.
- A detection must still be present after the travel delay for its serial command to be processed.
- Counts are not updated when the Arduino is unavailable.
- The bounding-box annotations are computed but the current display function shows the original belt crop rather than the annotated image.
- The application has no graphical configuration interface, configuration file, reconnect strategy, automated tests, or packaged release.
- Training code and datasets are not included.
- `export_frames.py` stores camera credentials directly in source code and should be changed before shared use.

## Troubleshooting

### The model file cannot be found

Run the application from the repository root. On Linux or macOS, replace the Windows-style path `model\best new v2.pt` with `model/best new v2.pt`.

### The camera stream does not open

- Verify the username, password, IP address, port, and stream path.
- Confirm that RTSP is enabled for the camera account.
- Ensure the computer can reach the camera and that the Wi-Fi network does not isolate clients.
- Test the URL in VLC or with FFmpeg.
- Ensure the installed OpenCV build includes FFmpeg support.

### No Arduino is found

- Confirm that the board is connected with a data-capable USB cable.
- Install the board's USB serial driver if required.
- Close the Arduino IDE serial monitor before starting the Python application.
- Check the available ports with `python -m serial.tools.list_ports`.
- Update `get_arduino_port()` if the Arduino is not the first enumerated serial device.

### Objects are detected but the servo moves at the wrong time

Calibrate both layers of timing: the `5.5`-second delay in `main.py` and `delayTime_1`/`delayTime_2` in `Servo.ino`. Their correct values depend on conveyor speed and the physical distance from the trigger line to each diverter position.

### Objects are missed or assigned incorrectly

- Improve lighting and reduce glare or motion blur.
- Recalibrate the crop and trigger line for the camera view.
- Confirm that the loaded model uses the expected class IDs.
- Adjust the `0.8` confidence threshold cautiously.
- Retrain the detector with images captured from the target conveyor setup if needed.

### CUDA is unavailable

The application automatically uses the CPU when `torch.cuda.is_available()` is false. If CUDA is expected, verify the NVIDIA driver and ensure that the installed PyTorch build matches the supported CUDA runtime.

## Additional Resources

- [Detailed setup videos on Google Drive](https://drive.google.com/drive/folders/1yxKsXGKaoJk8VRQk04IYU0DpA0V7p8SK?usp=sharing)
- [TP-Link Tapo RTSP setup guidance](https://www.tapo.com/vn/faq/51/)
- [Ultralytics documentation](https://docs.ultralytics.com/)
- [Supervision documentation](https://supervision.roboflow.com/)
- [Arduino Servo library](https://docs.arduino.cc/libraries/servo/)

## Authors

- [Nguyễn Mạnh Hưng](https://github.com/pikamanh)
- [Nguyễn Võ Hoàng Khang](https://github.com/khangkaka066)
- [Trần Văn Thuận](https://github.com/trankhacthuan)
