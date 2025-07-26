import sys
import transformers
import datasets
import torch

# Affiche le chemin de l'exécutable Python qui lance ce script
print(f"Chemin de l'interpréteur Python : {sys.executable}")
print("-" * 50)

# Affiche les versions des bibliothèques clés
print(f"Version de Transformers : {transformers.__version__}")
print(f"Version de Datasets : {datasets.__version__}")
print(f"Version de PyTorch : {torch.__version__}")
print("-" * 50)

# Vérifie si PyTorch utilise bien le GPU
print(f"PyTorch utilise le GPU (CUDA) : {torch.cuda.is_available()}")