# Label Studio for OCR with PaddleOCR

This guide provides instructions on setting up and using Label Studio for annotating OCR datasets, specifically tailored for integration with a PaddleOCR pipeline. It covers basic usage and an advanced pre-labeling workflow to accelerate your annotation process.

## 🚀 Quick Start

Get Label Studio up and running quickly with these steps:

1. **Install Label Studio:**
    Ensure you have Python 3.8 or newer installed.

    ```bash
    pip install label-studio
    ```

2. **Start Label Studio:**
    Run the following command in your terminal. Label Studio will typically open in your web browser at `http://localhost:8080`.

    ```bash
    label-studio
    ```

3. **Create an Account:**
    Upon your first launch, you'll be prompted to sign up. Create an account using your email address and a new password.

4. **Create a New Project:**
    On the project dashboard, click the `Create` button to start a new labeling project.

5. **Configure Project Details:**
    Give your project a meaningful name and, optionally, add a description.

6. **Select Labeling Template:**
    Click on `Labeling Setup` and choose the `Optical Character Recognition` template. This pre-configures Label Studio for OCR annotation tasks.

7. **Import Data:**
    Navigate to `Data Import` and upload the image files you wish to annotate.

8. **Save Your Project:**
    Click `Save` to finalize your project setup.

---

## 📸 Basic Usage

Once your project is set up, you can begin annotating your images:

1. **Open an Image:**
    From your project view, click on an image thumbnail to open it in the labeling interface.

2. **Annotate Bounding Boxes and Transcriptions:**
    Use the tools provided to draw bounding boxes around text regions and enter the corresponding ground truth transcription for each box.

3. **Understand Labeling Tools:**
    Refer to the image below for an explanation of the various labeling tools available within Label Studio:
    ![](./imgs/ls_basic_usage.png)

---

## 🤖 Pre-label with PaddleOCR

Leverage PaddleOCR to pre-label your data, significantly reducing manual annotation effort. **Ensure your PaddleOCR environment is correctly set up as per the main project README before proceeding.**

1. **Initial Sparse Annotation in Label Studio:**
    For each image you want to pre-label, randomly annotate _at least one bounding box_. You do _not_ need to add transcriptions at this stage; just the bounding box is sufficient to trigger the export.

2. **Export Data from Label Studio:**
    On your project page in Label Studio, click the `Export` button located at the top right. Select `YOLO with images` as the export format.

3. **Prepare Exported Data:**

    - Unzip the downloaded archive.
    - Move the `images` folder from the unzipped contents to the **root directory of this code repository**.
    - **Crucially, ensure the image folder is named `images/`**. If it has a different name, you will need to update the image path variable within the `inference.py` script to match.

4. **Run PaddleOCR Inference for Pre-labeling:**
    Execute the `inference.py` script. This script will use PaddleOCR to perform OCR on the images you just exported, generating preliminary bounding boxes and transcriptions.

    ```bash
    python inference.py
    ```

5. **Convert Pre-labels to Label Studio Format:**
    Run the `convert_prelabel.py` script. This script takes the output from PaddleOCR inference and converts it into a JSON format compatible with Label Studio.

    ```bash
    python convert_prelabel.py
    ```

    This will generate a file named `prelabeled.json`.

6. **Import Pre-labeled Data into Label Studio:**
    Go back to your Label Studio project. Use the "Import" functionality and import the `prelabeled.json` file you just generated. This will load the PaddleOCR-generated annotations into your project.

7. **Review and Correct Annotations:**
    Carefully go through the pre-labeled images in Label Studio. Correct any inaccurate bounding boxes, add missing ones, and ensure all transcriptions are precise. This is the quality control step.

8. **Export Final Annotations:**
    Once you have reviewed and corrected all annotations, export your project from Label Studio again, this time choosing the `JSON` format.

9. **Prepare Dataset for PaddleOCR Finetuning:**

    - Copy the exported JSON file to the `label_studio/` directory within this PaddleOCR code repository.
    - Rename the copied file to `labeled.json` (overwriting if a previous version exists).
    - Open and run the `prepare_dataset.ipynb` Jupyter notebook. This notebook will convert the Label Studio JSON format into the specific data format required by PaddleOCR for finetuning.

10. **Ready for Finetuning\!**
    Your dataset is now prepared and ready to be used for finetuning your PaddleOCR models as described in the main README's "Finetuning" section.

---
