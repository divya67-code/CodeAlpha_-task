# Deep Neural Emotion Intelligence 🧠🎙️

An advanced Speech-Based Human Emotion Recognition System utilizing a Hybrid Deep Learning architecture (CNN + BiLSTM + Attention) to predict emotions directly from raw audio signals.

## Project Features
- **Advanced Feature Extraction**: Uses Mel Spectrograms, MFCCs, Chroma, and Zero Crossing Rate via `librosa`.
- **Hybrid Architecture**: 
  - **1D CNN** for spatial frequency perception.
  - **BiLSTM** for learning temporal context from speech frames.
  - **Custom Attention Layer** to weigh emotionally significant frames.
- **Data Augmentation**: Robust training pipeline with dynamic pitch-shifting, time-stretching, and noise injection.
- **Interactive UI**: Real-time evaluation through a stunning Streamlit dashboard.

## Setup Instructions

1. **Install Dependencies**
Ensure you have Python 3.8+ installed. Then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

2. **Acquire Datasets**
Download the [RAVDESS](https://zenodo.org/record/1188976), [TESS](https://tspace.library.utoronto.ca/handle/1807/24487), or [EMO-DB](http://emodb.bilderbar.info/download/) datasets and place the WAV files into the `data/` directory.

3. **Train the Model**
Run the training script to parse data, train the model, and save the `.h5` model.
```bash
python src/train.py --data_dir data/ --epochs 50 --batch_size 32
```
This will generate `models/final_model.h5` and `models/label_encoder_classes.npy`.

4. **Run the Streamlit App**
Launch the interactive web application to upload audio or record your voice and see the predictions!
```bash
streamlit run app/main.py
```

## Structure
- `src/` - Core Python scripts to process, extract features, and train the model.
- `app/` - Streamlit UI logic.
- `models/` - Directory where model weights and labels are saved.
- `data/` - Put your `.wav` files here.
