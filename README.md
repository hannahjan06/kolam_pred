# 🌸 Kolam Cultural Heritage Web App

Welcome to the **Kolam Web App**, an interactive platform that preserves and promotes **Kolam**, a traditional South Indian floor art. This project combines **AI, cultural heritage, and interactive web design** in a single, easy-to-use prototype.  

---

## 🖼️ Features

| Feature | Description |
|---------|-------------|
| **Kolam Classification** | Upload a Kolam image and get AI-predicted motif/type using **EfficientNetB0**. |
| **Kolam Generation** | Generate new Kolam designs using AI. |
| **Chatbot** | Ask questions about Kolam history, symbolism, and techniques through an AI-powered chatbot. |
| **Draw Your Own Kolam** | Draw, color, and save personal Kolams using a canvas tool. |
| **Mood-based Recommendations** | Explore symmetric designs for relaxation or colorful ones for inspiration (art therapy). |

> 💡 **Tip:** Users can explore Kolam as a relaxing, creative experience — combining cultural learning with mood-based art therapy.  

---

## 🗂 Project Structure

```text
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
│
├── templates/
│   └── index.html
│
├── notebooks/              # for showing the training notebook
│   ├── preprocess_images.ipynb
│   └── train_model.ipynb
│
└── README.md
```
> **Note:** The `notebooks/` folder is optional and only for demo purposes; it shows how we preprocessed data and trained the model.  
> The running prototype only needs `app.py`, `.keras` model, and frontend files.

---

## ⚙️ Tech Stack

**Frontend:** HTML, CSS, JavaScript (canvas API for drawing)  
**Backend:** Python Flask  
**Machine Learning:** TensorFlow/Keras (EfficientNetB0 for classification, optional generative model)  
**Chatbot:** AI model via API (e.g., OpenAI GPT)  
**Deployment (Free options):**  
- Frontend → Vercel (static)  
- Backend → Render (Flask server)  

---

## 🚀 How to Run Locally

**Clone the repo:**
```bash
git clone <your-repo-url>
cd kolam_app
```
**Install dependencies**
```pip install -r requirements.txt
```
**Run the Flask app**
```python app.py
```
> By default, it will run at http://127.0.0.1:5000/.
> Open the frontend
> Open index.html in your browser (or use Flask’s template rendering). All features—upload, generate, chatbot, and drawing—will be functional.

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
