"""
Lightweight Voice Recognition Model
Target: < 1GB RAM, 5 speakers
Architecture: Mel spectrogram features + small CNN

Requirements:
    pip install torch torchaudio numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import numpy as np
from typing import Optional


class AudioPreprocessor:
    """
    Converts raw audio to mel spectrograms.
    Mel spectrograms capture voice characteristics efficiently.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,          # Reduced from typical 80 for memory
        n_fft: int = 400,          # ~25ms window at 16kHz
        hop_length: int = 160,     # ~10ms hop
        duration_sec: float = 2.0  # Fixed input duration
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration_sec = duration_sec
        self.target_length = int(sample_rate * duration_sec)
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # Convert to log scale (better for neural networks)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio file (supports mp4, wav, mp3, etc.)"""
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure fixed length for consistent input size"""
        length = waveform.shape[1]
        
        if length > self.target_length:
            # Trim from center (captures the word better than from start)
            start = (length - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        elif length < self.target_length:
            # Pad with zeros
            padding = self.target_length - length
            waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
        
        return waveform
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """Full preprocessing pipeline"""
        waveform = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize to [-1, 1] range
        mel_spec_db = (mel_spec_db + 40) / 40  # Rough normalization
        
        return mel_spec_db  # Shape: [1, n_mels, time_frames]


class LightweightVoiceNet(nn.Module):
    """
    Small CNN for speaker identification.
    
    Architecture designed for:
    - < 500KB model size
    - < 100MB inference memory
    - 5 speaker classification
    
    Total params: ~85K (vs millions in typical voice models)
    """
    
    def __init__(
        self,
        n_mels: int = 40,
        num_speakers: int = 5,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        
        # Convolutional feature extractor
        # Input: [batch, 1, n_mels, time_frames]
        self.conv_layers = nn.Sequential(
            # Block 1: [1, 40, T] -> [16, 20, T/2]
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2: [16, 20, T/2] -> [32, 10, T/4]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 3: [32, 10, T/4] -> [64, 5, T/8]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )
        
        # Global average pooling over time (makes model length-invariant)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer (useful for speaker verification later)
        self.embedding = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_speakers)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: Mel spectrogram [batch, 1, n_mels, time_frames]
            return_embedding: If True, also return the embedding vector
        
        Returns:
            logits: [batch, num_speakers]
            embedding (optional): [batch, embedding_dim]
        """
        # Feature extraction
        features = self.conv_layers(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # [batch, 64]
        
        # Embedding
        emb = self.embedding(pooled)
        
        # Classification
        logits = self.classifier(emb)
        
        if return_embedding:
            return logits, emb
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding (useful for verification)"""
        _, emb = self.forward(x, return_embedding=True)
        return emb


class VoiceRecognitionSystem:
    """
    Complete voice recognition system.
    Handles training, inference, and model persistence.
    """
    
    def __init__(
        self,
        num_speakers: int = 5,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = AudioPreprocessor()
        self.model = LightweightVoiceNet(num_speakers=num_speakers).to(self.device)
        self.speaker_names: list[str] = []
    
    def prepare_dataset(self, data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load dataset from directory structure:
        
        data_dir/
            speaker_name/
                word1/
                    word1_speaker_1.m4a
                    word1_speaker_2.m4a
                    ...
                word2/
                    ...
            another_speaker/
                ...
        
        Returns:
            features: [N, 1, n_mels, time_frames]
            labels: [N]
        """
        data_path = Path(data_dir)
        features = []
        labels = []
        
        # Get speaker directories
        speaker_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        self.speaker_names = [d.name for d in speaker_dirs]
        
        print(f"Found {len(speaker_dirs)} speakers: {self.speaker_names}")
        
        audio_extensions = ['.mp4', '.wav', '.mp3', '.m4a', '.flac']
        
        for speaker_idx, speaker_dir in enumerate(speaker_dirs):
            audio_files = []
            
            # Check for word subfolders
            word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
            
            if word_dirs:
                # Nested structure: speaker/word/files
                for word_dir in word_dirs:
                    for f in word_dir.iterdir():
                        if f.suffix.lower() in audio_extensions:
                            audio_files.append(f)
            else:
                # Flat structure: speaker/files
                for f in speaker_dir.iterdir():
                    if f.suffix.lower() in audio_extensions:
                        audio_files.append(f)
            
            audio_files = sorted(audio_files)
            print(f"  {speaker_dir.name}: {len(audio_files)} samples")
            
            for audio_file in audio_files:
                try:
                    mel_spec = self.preprocessor(str(audio_file))
                    features.append(mel_spec)
                    labels.append(speaker_idx)
                except Exception as e:
                    print(f"    Warning: Failed to load {audio_file.name}: {e}")
        
        features = torch.stack(features)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return features, labels
    
    def train(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """Train the model on the dataset"""
        
        # Load data
        print("Loading dataset...")
        features, labels = self.prepare_dataset(data_dir)
        
        # Train/val split
        n_samples = len(features)
        indices = torch.randperm(n_samples)
        n_val = int(n_samples * validation_split)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_features = features[train_indices].to(self.device)
        train_labels = labels[train_indices].to(self.device)
        val_features = features[val_indices].to(self.device)
        val_labels = labels[val_indices].to(self.device)
        
        print(f"Training samples: {len(train_features)}, Validation: {len(val_features)}")
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Data augmentation (simple time masking)
        def augment(x):
            if self.model.training and torch.rand(1).item() > 0.5:
                # Mask random time segment
                t = x.shape[-1]
                mask_len = int(t * 0.1)
                start = torch.randint(0, t - mask_len, (1,)).item()
                x = x.clone()
                x[..., start:start+mask_len] = 0
            return x
        
        # Training loop
        best_val_acc = 0
        print("\nStarting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            
            # Mini-batch training
            perm = torch.randperm(len(train_features))
            for i in range(0, len(train_features), batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_x = train_features[batch_idx]
                batch_y = train_labels[batch_idx]
                
                # Augment
                batch_x = augment(batch_x)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_x)
                train_correct += (logits.argmax(1) == batch_y).sum().item()
            
            scheduler.step()
            
            train_loss /= len(train_features)
            train_acc = train_correct / len(train_features)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_features)
                val_loss = criterion(val_logits, val_labels).item()
                val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save('best_model.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
                      f"Val Loss={val_loss:.4f}, Acc={val_acc:.2%}")
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
        self.load('best_model.pt')
    
    def predict(self, audio_path: str) -> tuple[str, float, dict]:
        """
        Predict speaker from audio file.
        
        Returns:
            speaker_name: Predicted speaker
            confidence: Prediction confidence (0-1)
            all_probs: Dict of speaker -> probability
        """
        self.model.eval()
        
        with torch.no_grad():
            mel_spec = self.preprocessor(audio_path)
            mel_spec = mel_spec.unsqueeze(0).to(self.device)  # Add batch dim
            
            logits = self.model(mel_spec)
            probs = F.softmax(logits, dim=1).squeeze()
            
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            
            all_probs = {
                name: probs[i].item() 
                for i, name in enumerate(self.speaker_names)
            }
        
        return self.speaker_names[pred_idx], confidence, all_probs
    
    def get_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding for verification tasks"""
        self.model.eval()
        
        with torch.no_grad():
            mel_spec = self.preprocessor(audio_path)
            mel_spec = mel_spec.unsqueeze(0).to(self.device)
            embedding = self.model.get_embedding(mel_spec)
        
        return embedding.cpu().numpy().squeeze()
    
    def verify_speaker(
        self, 
        audio_path: str, 
        reference_embedding: np.ndarray,
        threshold: float = 0.7
    ) -> tuple[bool, float]:
        """
        Verify if audio matches a reference speaker embedding.
        
        Returns:
            is_match: True if similarity > threshold
            similarity: Cosine similarity score
        """
        test_embedding = self.get_speaker_embedding(audio_path)
        
        # Cosine similarity
        similarity = np.dot(test_embedding, reference_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(reference_embedding)
        )
        
        return similarity > threshold, float(similarity)
    
    def save(self, path: str):
        """Save model and metadata"""
        torch.save({
            'model_state': self.model.state_dict(),
            'speaker_names': self.speaker_names,
            'num_speakers': self.model.num_speakers
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.speaker_names = checkpoint['speaker_names']
        print(f"Model loaded from {path}")
    
    def memory_usage(self) -> dict:
        """Estimate memory usage"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        return {
            'model_params': f"{param_size / 1024:.1f} KB",
            'model_buffers': f"{buffer_size / 1024:.1f} KB",
            'total_model': f"{(param_size + buffer_size) / 1024:.1f} KB",
            'estimated_inference_ram': "~50-100 MB"
        }


def print_model_info(model: LightweightVoiceNet):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("Model Summary")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB (float32)")
    print("="*50)


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = VoiceRecognitionSystem(num_speakers=5)
    
    # Print model info
    print_model_info(system.model)
    print("\nMemory usage:")
    for k, v in system.memory_usage().items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*50)
    print("Usage Instructions")
    print("="*50)
    print("""
1. Organize your data:
   data/
       person_1/
           hello_1.mp4
           hello_2.mp4
           open_1.mp4
           ...
       person_2/
           ...

2. Train:
   system = VoiceRecognitionSystem(num_speakers=5)
   system.train('data/', epochs=50)
   system.save('voice_model.pt')

3. Predict:
   system.load('voice_model.pt')
   speaker, confidence, probs = system.predict('test_audio.mp4')
   print(f"Speaker: {speaker} ({confidence:.1%} confidence)")
""")
