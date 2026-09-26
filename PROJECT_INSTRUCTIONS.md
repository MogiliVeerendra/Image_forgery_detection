# Image Forgery Detection --- Project Instructions

This guide explains how to download the project from GitHub, extract it,
open it in Visual Studio Code, set up Python, and run the web
application in your browser.

## 1. Requirements

Install these tools before starting:

-   **GitHub account or a web browser** --- to download the project.
-   **Python 3.10** --- the project environment was set up with Python
    3.10.
-   **Visual Studio Code (VS Code)** --- to open and edit the code.
-   **Python extension for VS Code** --- for Python support.

Useful download pages:

-   Python: https://www.python.org/downloads/
-   VS Code: https://code.visualstudio.com/
-   Project repository:
    https://github.com/MogiliVeerendra/Image_forgery_detection

During Python installation on Windows, enable **Add Python to PATH** if
that option is shown. If the `python` command does not work, use the
`py -3.10` launcher in the commands below.

## 2. Download the project from GitHub

1.  Open the repository:
    https://github.com/MogiliVeerendra/Image_forgery_detection
2.  Click the green **Code** button.
3.  Select **Download ZIP**.
4.  Wait for the ZIP file to finish downloading.

The downloaded file will usually be in your **Downloads** folder.

### Alternative: download using Git

If Git is installed, open a terminal in the location where you want the
project and run:

``` powershell
git clone https://github.com/MogiliVeerendra/Image_forgery_detection.git
```

This creates a folder named `Image_forgery_detection`.

## 3. Extract the ZIP file

If you downloaded the ZIP:

1.  Open File Explorer and go to **Downloads**.
2.  Find `Image_forgery_detection-main.zip` (the exact name may vary).
3.  Right-click the ZIP file.
4.  Choose **Extract All...**.
5.  Choose a destination and click **Extract**.

For the steps below, this guide assumes the project folder is:

``` text
C:\image_forgery_detection
```

Your folder can be somewhere else. If so, use your actual folder path in
the commands.

**Important:** Open the extracted project folder---the folder containing
`requirements.txt`, `models`, and `web_app`---not the ZIP file itself or
an extra outer folder.

## 4. Open the project in VS Code

### Option A: using VS Code

1.  Open Visual Studio Code.
2.  Select **File → Open Folder...**.
3.  Navigate to `C:\image_forgery_detection`.
4.  Select the project folder and click **Select Folder**.
5.  If VS Code asks whether you trust the folder, confirm only if you
    trust the source.

### Option B: using PowerShell

Open PowerShell and run:

``` powershell
cd C:\image_forgery_detection
code .
```

If `code` is not recognized, use Option A.

## 5. Open the VS Code terminal

In VS Code, select:

**Terminal → New Terminal**

A terminal panel should appear at the bottom.

Check that the terminal is in the project root. Run:

``` powershell
pwd
```

It should show the project directory, for example:

``` text
C:\image_forgery_detection
```

If it does not, change to the project directory:

``` powershell
cd C:\image_forgery_detection
```

## 6. Create a Python virtual environment

In the VS Code PowerShell terminal, run:

``` powershell
py -3.10 -m venv .venv
```

This creates an isolated Python environment in the `.venv` folder.

### Activate the environment

``` powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal prompt should begin with:

``` text
(.venv)
```

If PowerShell blocks activation because scripts are disabled, run this
command for the current terminal session:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

``` powershell
.\.venv\Scripts\Activate.ps1
```

This changes the execution policy only for the current PowerShell
process.

### If Python 3.10 is not found

Check installed Python versions:

``` powershell
py -0p
```

If Python 3.10 is not listed, install Python 3.10 and reopen the
terminal.

## 7. Select the Python interpreter in VS Code

1.  Press `Ctrl + Shift + P`.
2.  Search for **Python: Select Interpreter**.
3.  Select the interpreter inside this project's `.venv` folder. It
    should look similar to:

``` text
.\.venv\Scripts\python.exe
```

This helps VS Code use the same environment as the terminal.

## 8. Install the project dependencies

Make sure `.venv` is activated, then run:

``` powershell
python -m pip install --upgrade pip
```

Install the packages listed in the project's requirements file:

``` powershell
python -m pip install -r requirements.txt
```

If the app reports that Streamlit is missing, install it in the active
environment:

``` powershell
python -m pip install streamlit
```

If TensorFlow or another package fails to install, check that the
selected Python version is compatible with the package versions
specified in `requirements.txt`.

## 9. Check that the model file exists

The web app expects the trained model at:

``` text
models\forgery_model.h5
```

From the project root, check it with:

``` powershell
Test-Path .\models\forgery_model.h5
```

If the result is `True`, the file exists.

If the result is `False`, the model file is missing. Obtain the required
model file from the project's author or the project's documented
model-download location. The app cannot make predictions without it.

## 10. Check the Python file for syntax errors

Before launching the web app, run:

``` powershell
python -m py_compile web_app\app.py
```

If the command produces no output, Python found no syntax errors in that
file.

If it reports a syntax error, read the file path and line number in the
error, then correct the code before continuing.

## 11. Run the project in a live web browser

From the project root, with `.venv` activated, run:

``` powershell
python -m streamlit run web_app\app.py
```

Streamlit will start a local web server. The terminal should display a
**Local URL**, usually:

``` text
http://localhost:8501
```

Hold `Ctrl` and click the URL in the terminal, or copy it into your
browser.

The application runs locally on your computer. Keep the terminal open
while using it. To stop the server, return to the terminal and press:

``` text
Ctrl + C
```

### If port 8501 is already in use

Run the app on another port:

``` powershell
python -m streamlit run web_app\app.py --server.port 8502
```

Then open:

``` text
http://localhost:8502
```

## 12. Use the application

1.  Open the local web address in your browser.
2.  Click the image uploader.
3.  Select a `.jpg`, `.jpeg`, or `.png` image.
4.  Review the predicted label: **Authentic** or **Tampered**.
5.  Review the confidence and both class probabilities.
6.  Review the Grad-CAM heatmap, overlay, and highlighted high-attention
    regions.
7.  Adjust the localization threshold if needed.
8.  Expand **Model Output (Debug)** to inspect model output details.

## 13. Understanding the prediction and localization

The saved model has a single sigmoid output. In the current app, the
output is interpreted as the **Tampered probability**:

-   Tampered probability at or above `0.50` → **Tampered**
-   Tampered probability below `0.50` → **Authentic**
-   Authentic probability = `1 - Tampered probability`

For example, if Tampered is `28.25%`, Authentic is `71.75%`, and the
predicted class is Authentic.

Grad-CAM uses the model layer `conv5_block3_out` to create a heatmap. It
highlights regions that influenced the model's prediction. **It does not
prove that those exact pixels were forged.** A model can also
misclassify an image, so verify results using images with known labels.

## 14. Test with authentic and forged images

Test the app with images whose ground truth you know:

-   **Authentic:** an original, unedited image.
-   **Tampered:** an image intentionally modified, for example by
    copying an object from another image or inserting/removing an
    object.

Keep a record of the ground truth, prediction, confidence, and
localization output. Do not assume every edited image will be detected;
performance depends on the training data and model.

For a proper evaluation, test a separate set of labeled images and
calculate metrics such as accuracy, precision, recall, F1-score, and a
confusion matrix.

## 15. Common problems and fixes

### `python` is not recognized

Use the Python launcher:

``` powershell
py -3.10 --version
```

When creating the environment, use:

``` powershell
py -3.10 -m venv .venv
```

After activating the environment, use `python` for project commands.

### `streamlit` is not recognized

Use:

``` powershell
python -m streamlit run web_app\app.py
```

If Streamlit is not installed:

``` powershell
python -m pip install streamlit
```

### `ModuleNotFoundError`

Activate `.venv`, then install dependencies:

``` powershell
python -m pip install -r requirements.txt
```

Install any specifically missing package into the same environment.

### Model file not found

Confirm that this file exists:

``` text
models\forgery_model.h5
```

Run the `Test-Path` command in Section 9.

### App opens, but the prediction seems incorrect

A prediction is not a guarantee. Check that the image is appropriate for
the model and that preprocessing matches the preprocessing used during
training. Test multiple known authentic and tampered images.

### Grad-CAM does not show the exact edited pixels

Grad-CAM is an explanation of model influence, not a pixel-level forgery
mask. Exact localization requires suitable pixel-mask training data and
a segmentation model.

## 16. Project structure

The project should contain files and folders similar to:

``` text
image_forgery_detection/
├── models/
│   └── forgery_model.h5
├── src/
├── web_app/
│   └── app.py
├── datasets/
├── requirements.txt
├── README.md
└── PROJECT_INSTRUCTIONS.md
```

The exact contents may differ depending on the version of the
repository.

## 17. Quick start --- commands in order

After Python 3.10, VS Code, and the project have been downloaded:

``` powershell
cd C:\image_forgery_detection
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install streamlit
python -m py_compile web_app\app.py
python -m streamlit run web_app\app.py
```

Then open the Local URL printed in the terminal, usually
`http://localhost:8501`.

If you already created `.venv`, do not create it again; activate it and
continue from dependency installation.
