import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tqdm import tqdm
# Import Axes3D for 3D projection
from mpl_toolkits.mplot3d import Axes3D 

class AudioProcessor:
    def __init__(self, model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval()

    def _prepare_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        return self.extractor(audio, sampling_rate=sr, return_tensors="pt")

    def extract_embeddings(self, file_path):
        """Returns a single vector representing the entire audio file."""
        inputs = self._prepare_audio(file_path)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Shape: [1, 1214, 768] -> [768]
        return outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()

    def visualize_space(self, file_list, labels=None):
        """
        Processes a list of files and plots them in 3D space.
        """
        features = []
        print(f"Extracting Embeddings From: {file_list}")
        for f in tqdm(file_list):
            features.append(self.extract_embeddings(f))
        
        # 1. Change PCA components to 3
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(features)

        # 2. Setup 3D figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3. Plot with 3 dimensions
        scatter = ax.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1], 
            reduced_data[:, 2], 
            c=range(len(file_list)) if labels is None else labels,
            cmap='viridis',
            s=50
        )
        
        print(f"Annotating PCA")
        for i in tqdm(range(len(file_list))):
            txt = file_list[i].split('/')[-1]
            # 4. Add text in 3D space
            ax.text(
                reduced_data[i, 0], 
                reduced_data[i, 1], 
                reduced_data[i, 2], 
                txt, 
                fontsize=8
            )

        ax.set_title("Audio Feature Space (3D PCA)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        
        plt.savefig("pca_3d.png")
        plt.close()
