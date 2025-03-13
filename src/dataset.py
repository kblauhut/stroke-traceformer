from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.tokens import CurveTokenizer
import torch
import json
import os
from src.config import ModelConfig

class StrokeDataset(Dataset):
    def __init__(self, config: ModelConfig, dataFolderPath: str, tokenizer: CurveTokenizer):
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_length = config.context_length
        self.img_size = config.img_size

        for folder in os.listdir(dataFolderPath):
            absolute_path = os.path.abspath(os.path.join(dataFolderPath, folder))
            if not os.path.isdir(absolute_path):
                continue

            png_path = os.path.join(absolute_path, "curves.png")
            json_path = os.path.join(absolute_path, "curves.json")
            if not os.path.exists(png_path) or not os.path.exists(json_path):
                continue

            png = Image.open(png_path).convert("RGB")
            with open(json_path, "r") as f:
                json_data = json.load(f)

            self.data.append([png, json_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        [png, json_data] = self.data[idx]
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # Resize to model's input size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(png)
        tokenized_json = self.tokenizer.tokenize_commands_pad(json_data, lenght=self.context_length)
        return image_tensor, torch.tensor(tokenized_json)
