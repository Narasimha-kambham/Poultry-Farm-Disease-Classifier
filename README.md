# Poultry Disease Classification

This project is a machine learning application for classifying poultry diseases from images. It uses a deep learning model to predict whether a poultry image is healthy or affected by a specific disease.

## Features
- Upload poultry images for disease classification
- Deep learning model trained on poultry disease dataset
- Web interface for easy usage

## Project Structure
- `app.py` : Main Flask application
- `src/` : Source code (data processing, model, training, utilities)
- `static/uploads/` : Uploaded images
- `templates/` : HTML templates for the web interface
- `poultry_disease_model.keras` : Trained model file
- `requirements.txt` : Python dependencies

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Open your browser and go to:**
   [http://localhost:5000](http://localhost:5000)

## Usage
- Upload a poultry image using the web interface.
- The model will predict and display the disease class.

## Notes
- Make sure you have the trained model file (`poultry_disease_model.keras`) in the project directory.
- For retraining or modifying the model, refer to the scripts in the `src/` folder.

## License
This project is for educational purposes.
```

You can copy and save this as `README.md` in your project root directory.
        