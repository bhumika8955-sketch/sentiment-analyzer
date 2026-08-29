# 🤖 AI-Powered Product Review Sentiment Classifier System

An end-to-end Machine Learning web application that leverages Natural Language Processing (NLP) to dynamically categorize user product feedback into positive or negative sentiments in real-time.

## 🔗 Live Project Links
* **Live Deployment App:** https://sentiment-analyzer-gajqqzpjqya2fpyrhqeaad.streamlit.app/
* **Development Workspace:** https://colab.research.google.com/drive/1qrWY4MqTjdi9noRKv_QFga9L8naHsfz1?usp=sharing

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3
* **AI Engine & Pipeline:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`)
* **Data Processing:** Pandas & NumPy
* **Cloud Infrastructure:** Streamlit Community Cloud
* **SDLC Design Assistant:** IBM Bob

## ⚙️ How It Works (Technical Overview)
1. **Data Management:** Modeled a custom local dataset pipeline containing balanced sentiment text matrices to maintain local execution stability.
2. **Feature Engineering:** Implemented **TF-IDF Vectorization** (*Term Frequency-Inverse Document Frequency*) to map unstructured text sentences into numerical feature arrays based on word weights.
3. **Model Classifier:** Trained a mathematical **Logistic Regression Model** using the calculated vector matrix to predict binary text sentiment classes.
4. **Cloud UI Layer:** Implemented a streamlined Python UI script to host the backend machine learning model globally as a serverless web app interface.

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn streamlit
   ```
3. Launch the app dashboard:
   ```bash
   streamlit run app.py
   ```
