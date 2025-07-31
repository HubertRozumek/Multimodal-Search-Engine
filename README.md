# Multimodal Search Engine with CLIP and RAG

Advanced product search system using **CLIP**, **FAISS**, and **RAG** with text/image search capabilities. Supports Fashion-MNIST and custom datasets.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)

## Key Features
- **Multimodal search**: Text, image, or hybrid queries
- **Advanced architecture**: CLIP embeddings + FAISS vector DB + RAG
- **Visual analytics**: Embedding visualization (t-SNE/PCA) and search metrics
- **Dataset support**: Fashion-MNIST and custom datasets
- **GPU acceleration**: MPS (Apple Silicon) / CUDA (NVIDIA)

## Quick Start
```bash
git clone https://github.com/HubertRozumek/Multimodal-Search-Engine.git
cd multimodal-search-engine
pip install -r requirements.txt
```
# Run with Fashion-MNIST
```bash
python main.py --dataset fashion-mnist
```
# Run with custom dataset
```bash
python main.py --dataset custom --dataset-path "./my_dataset"
```

Access web UI at http://localhost:7860

# Installation
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

# Custom folder structure:
```bash
my_dataset/
├── category1/
│   ├── img1.jpg
│   └── img2.jpg
└── category2/
    └── img3.jpg
```