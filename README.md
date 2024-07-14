Medical Information Extraction Automation
This project automates the extraction of medical information from images and texts using machine learning techniques. It integrates Optical Character Recognition (OCR), Named Entity Recognition (NER), and pre-trained Large Language Models (LLMs). The dataset for training and evaluation is sourced through web scraping from medical information sources.

Live Demo
Watch the live demo: Final_video_Aventus.mp4

Workflow
Input
Images: Medical documents, prescriptions, labels.
Text: Medical texts for NER and LLM processing.
Table of Contents
Overview
Features
Installation
Usage
Dataset
Contributing
License
Overview
This project automates the extraction of medical information from images and text sources using machine learning models. It aims to enhance the efficiency and accuracy of handling medical data, particularly in digitalizing prescriptions and extracting vital information from medical documents.

Features
OCR Integration: Convert medical images to text for digital processing.
NER Model: Named Entity Recognition for accurate medicine name extraction.
LLM Integration: Utilizes Mixtral-8x7B for advanced natural language processing tasks.
Web Scraping: Collects and processes medical data from diverse sources for training.
Installation
Prerequisites
Python 3.6 or higher
pip package manager
Git
Clone the repository
bash
Copy code
git clone <repository-url>
cd <repository-directory>
Create a virtual environment (optional but recommended)
bash
Copy code
python -m venv venv
Activate the virtual environment
On Windows:
bash
Copy code
venv\Scripts\activate
On macOS/Linux:
bash
Copy code
source venv/bin/activate
Install dependencies
bash
Copy code
pip install -r requirements.txt
Usage
To run the application:

bash
Copy code
streamlit run Main.py
Follow the prompts and instructions in the application for image/text input and processing.

Dataset
The dataset used in this project is collected through web scraping from reliable medical sources. It includes a comprehensive collection of medical texts and drug-related information for training and evaluation.

Contributing
Contributions are welcome! If you want to contribute to this project, please fork the repository and create a pull request with your proposed changes. Feel free to open issues for bugs, feature requests, or general discussions.

License
This project is licensed under the MIT License.
