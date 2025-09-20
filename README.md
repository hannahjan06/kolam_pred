Kolam Cultural Heritage Web App
🌸 Overview

Welcome to the Kolam Web App! This interactive platform preserves and promotes Kolam, a traditional South Indian floor art. Our project allows users to:

Classify Kolam designs: Upload a photo of a Kolam, and our AI identifies its type using a pre-trained EfficientNetB0 model.

Generate new Kolams: Press a button and create a unique Kolam design using a generative AI model.

Ask the Kolam Chatbot: Learn about Kolam history, symbolism, and techniques through an AI-powered chatbot.

Draw your own Kolam: Use the canvas tool to draw and save your own Kolam creations.

Experience mood-based therapy: Explore symmetric or colorful patterns depending on your mood—helping users relax or get inspired.

This project combines AI, cultural heritage, and interactive web design in a single, easy-to-use prototype.

🗂 Project Structure
/kolam_app
│
├── app.py                  # Flask backend
├── requirements.txt        # Python packages
├── Procfile                # For deployment
│
├── models/                 
│   └── kolam_classifier.keras   # Pre-trained EfficientNetB0 model
│
├── static/
│   ├── css/
│   │   └── stylesheet.css
│   ├── js/
│   │   └── script.js
│   └── images/             # Any static images/icons
│
├── templates/
│   └── index.html
│
├── notebooks/              # Optional: show workflow to judges
│   ├── preprocess_images.ipynb   # Convert images to matrices
│   └── train_model.ipynb         # Training and testing
│
└── README.md


Note: The notebooks/ folder is optional and only for demo purposes; it shows how we preprocessed data and trained the model. The running prototype only needs app.py, .keras model, and frontend files.

⚙️ Tech Stack

Frontend: HTML, CSS, JavaScript (canvas API for drawing)

Backend: Python Flask

Machine Learning: TensorFlow/Keras (EfficientNetB0 for classification, optional generative model)

Chatbot: AI model via API (e.g., OpenAI GPT)

Deployment (Free options):

Frontend: Vercel (static)

Backend: Render (Flask server)

🚀 How to Run Locally

Clone the repo

git clone <your-repo-url>
cd kolam_app


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


By default, it will run at http://127.0.0.1:5000/.

Open the frontend
Open index.html in your browser (or use Flask’s template rendering). All features—upload, generate, chatbot, and drawing—will be functional.

🖌 Features in Detail
Feature	Description
Kolam Classification	Upload a Kolam image, get AI-predicted motif/type.
Kolam Generation	Generate new Kolam designs using AI.
Chatbot	Ask questions about Kolam, history, and technique.
Draw Your Own Kolam	Draw, color, and save personal Kolams.
Mood-based Recommendations	View symmetric designs for relaxation or colorful ones for inspiration.
📦 Notebooks (Optional)

For judges who want to see our workflow:

preprocess_images.ipynb → Shows how we converted images to matrix form for model input.

train_model.ipynb → Contains model training, testing, and saving .keras files.

These notebooks are not required for running the live demo.

💡 Extra Notes

Model files: Only kolam_classifier.keras is needed for running the prototype.

Data: Original datasets are not included; the prototype runs with the saved model.

Extensibility: You can add a generative Kolam model, enhanced chatbot, or user login features later.

📜 References

KolamNet: Kolam classification using EfficientNetB0

Generative AI for Rangoli/Kolam patterns

Art therapy benefits of coloring mandala/symmetric patterns

Digitization of cultural heritage – Ministry of Culture

👩‍💻 Authors

Your Team Name / Hackathon Team Members