from src.dataset import StrokeDataset
from src.model import TransformerModel
from src.config import ModelConfig, DEVICE
from src.train_model import train_model
from src.tokens import CurveTokenizer

config = ModelConfig()
tokenizer = CurveTokenizer()
dataset = StrokeDataset(config, "data/", tokenizer)
model = TransformerModel(tokenizer.get_token_count(), config, DEVICE)
train_model(model, dataset, DEVICE, 20, 16)
