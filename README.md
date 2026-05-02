# FYP_CCDS25-0121: Multimodal SER with Llama-3

**A robust Multimodal Speech Emotion Recognition (SER) framework that fine-tunes Llama-3 to process audio features alongside textual transcription.**

> **Key Result:** Achieved **72.0% Macro-F1** on IEMOCAP (SOTA-competitive) and **25.1% Macro-F1** in MSP-Podcast (Fine-Grained Emotions).

![Pipeline Overview](asset/SpeechCueLLM_diagram_proposed.png)
*Figure 1: End-to-end pipeline illustrating the audio frontend, feature extraction (OpenSMILE), and the LLM inference engine.*

---

## Key Innovations
This project builds upon the **SpeechCueLLM** architecture, introducing significant engineering optimizations for robustness and efficiency:

* **Structured Prompting Strategy:** Replaced unstructured narrative prompts with a rigorous **JSON-style structured format**. This improved token efficiency by **~30%** and reduced model hallucination.
* **Enhanced Feature Extraction:** Integrated the **OpenSMILE** toolkit to extract prosodic and spectral features, enriching the context provided to the LLM beyond standard ASR.
* **Emotional Dimensions Integration:** Integrated emotional dimensions (valence, arousal, dominance) into the feature set, enriching the context provided to the LLM.
* **Dynamic Feature Masking:** Implemented a training strategy that randomly masks specific modalities (audio/text) during fine-tuning. This forces the model to learn robust representations, preventing over-reliance on a single data stream.
* **Automated Audio Preprocessing Model:** Developed a dedicated `audio_preprocess_model` that automates the pipeline from raw `.wav` to various metadata, decoupled from the LLM environment.

---

## Architecture

### 1. Audio Preprocessing Model (`/audio_preprocess_model`)
A standalone module designed to handle raw signal processing.
* **ASR Integration:** Uses OpenAI Whisper to generate high-fidelity transcripts.
* **VAD Regressor:** Predicts emotional dimensions (valence, arousal, dominance) of raw audio.
* **Gender Classifier:** Classifies the gender of speakers.
* **Feature Extractor:** Extracts acoustic features (eGeMaps) from audio.

### 2. The LLM Engine (Llama-3)
The core reasoning engine takes the processed inputs and generates emotion predictions.

<div align="center">
  <img src="asset/speechcuellm_prompt_proposed.png" width="85%" alt="Proposed Prompt">
  <p><em>Figure 2: New Structured Prompt format.</em></p>
</div>

---

## Datasets & Compliance

This framework supports analysis on **IEMOCAP** and **MSP-Podcast**.

> **Licensing Note:**
> This repository contains **preprocessing code only**. It does **not** host the raw audio data or derived features, in strict compliance with the dataset licensing agreements.

### IEMOCAP
* **Status:** Preprocessing pipeline included.
* **Access:** Request directly from [USC SAIL Lab](https://sail.usc.edu/iemocap/iemocap_release.htm).


### MSP-Podcast
* **Status:** Preprocessing pipeline included.
* **Access:** Request directly from [UTD MSP Lab](https://www.lab-msp.com/MSP/MSP-Podcast.html).

---

## Repo Structure
* `repo_home/data/iemocap_dataset`, `repo_home/data/msp_dataset` - Put your raw datasets here. Intermediate and final datasets will be saved in data/{dataset}_dataset. (Ignored by Git)
* `repo_home/checkpoints/` - Model weights and training logs of audio_preprocessing_model are saved here. (Ignored by Git)
* `repo_home/experiments/` - LLM Model weights and DeepSpeed logs are saved here. (Ignored by Git)
* `repo_home/venv/` - Your virtual environments will live here. (Ignored by Git)
* `repo_home/src/` - All pipeline execution scripts and Python source code.

---

## Environment Setup

You must create two separate virtual environments: one for the Audio pipeline and one for the LLM pipeline.

### System Level Dependencies
CUDA Version: 12.4
GCC/G++ Version: 13.1.0 
### 1a. Setup the Audio Environment
conda create -n venv_audio python=3.10.20
conda activate venv_audio
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_audio.txt


### 1b. Setup the LLM Environment
conda create -n venv_llm python=3.10.20
conda activate venv_llm
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install ninja==1.13.0
pip install -r requirements_llm.txt

### 2. create .env file
".env" file save in repo_home/src with the following line
HF_ACCESS_TOKEN="your_huggingface_token_here" # Allows login to hf to obtain llama model weight

## Examples
### Run dataset preprocessing code to prepare csv for audio pipeline (automatically uses venv_audio)
bash src/preprocess_dataset.sh

### Run the audio pipeline (automatically uses venv_audio)
bash src/main.sh

### Finetune using LoRA on the IEMOCAP dataset (automatically uses venv_llm)
bash src/main_llm.sh "iemocap" "lora_training"

### Run inference/evaluation only on the MSP dataset (automatically uses venv_llm)
bash src/main_llm.sh "msp" "inference"

---

## Acknowledgements & Credits
This project is an evolution of the SpeechCueLLM framework.

Original Architecture: SpeechCueLLM (https://github.com/zehuiwu/SpeechCueLLM)

Modifications: This implementation extends the original work by integrating OpenSMILE, introducing Dynamic Masking, and refactoring the prompting strategy for Llama-3 optimization.

