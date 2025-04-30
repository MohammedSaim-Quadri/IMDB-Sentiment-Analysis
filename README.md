# 🎬 IMDB Movie Review Sentiment Analysis

A deep learning-based web application that classifies movie reviews from the IMDB dataset into **positive** or **negative** sentiments using an RNN (SimpleRNN) model. Built using TensorFlow/Keras and deployed with Streamlit, the app allows users to input reviews and instantly get sentiment predictions.

---

## 🌐 Live Demo

> ⚙️ [View the live app](https://imdb-sentiment-analysis-fcnuccxn4j5yrvghbmdfmz.streamlit.app/)

---

## 🚀 Project Overview

The project demonstrates the power of Recurrent Neural Networks (RNNs) for Natural Language Processing (NLP), specifically sentiment classification. Using the popular IMDB dataset, the app predicts whether a given movie review expresses a **positive** or **negative** sentiment.

---

## 🧠 Model Architecture

- **Embedding Layer**: Transforms input word indices into dense vector representations.
- **SimpleRNN Layer**: Captures sequential dependencies between words.
- **Dense Output Layer**: Sigmoid activation for binary classification.
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

---

## 📂 Project Structure
```bash
├── imdb_rnn_model.h5 # Trained RNN model
├── main.py # Streamlit app entry point
├── prediction.ipynb # Model loading and prediction notebook
├── rnn.ipynb # RNN training notebook
├── embedding.ipynb # Word embedding exploration
├── requirements.txt # List of dependencies
├── LICENSE # GNU GPL license
└── README.md # Project documentation
```

---

## ✨ Features

- Input your own movie review and get real-time sentiment prediction
- Uses pre-trained RNN model for fast and accurate classification
- Fully interactive frontend built with Streamlit
- Real-world NLP preprocessing with IMDB word index and Keras tokenization

---

## 🔍 How It Works

1. The user submits a movie review.
2. The text is tokenized using the **IMDB word index** and padded to length 500.
3. The preprocessed text is passed to a pre-trained RNN.
4. The app returns:
   - **Sentiment**: Positive or Negative
   - **Prediction Score**: A float between 0 and 1

---

## 📈 Dataset

- **Source**: [IMDB Dataset of 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Labels**: Binary (0 = Negative, 1 = Positive)
- **Vocabulary Size**: Top 10,000 words retained
- **Preprocessing**:
  - Integer encoding via `keras.datasets.imdb.get_word_index()`
  - Padding to `maxlen=500` using `pad_sequences`

---

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/mohammedsaim-quadri-imdb-sentiment-analysis.git
cd mohammedsaim-quadri-imdb-sentiment-analysis
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run main.py
```

##🧪 Sample Usage
Try entering a review like:
```
"This movie was absolutely amazing! The plot and acting were top-notch."
```

You'll get output like:

```yaml
Sentiment: Positive
Prediction Score: 0.9123
```

## 📌 Dependencies
- TensorFlow
- NumPy
- Pandas
- Streamlit

All dependencies are listed in requirements.txt.

## 📜 License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more information.

## 👨‍💻 Author
Mohammed Saim Ahmed Quadri
📫 [LinkedIn](https://www.linkedin.com/in/msaquadri)
📧 mohammedsaimquadri@gmail.com
