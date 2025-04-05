import torch
import torch.onnx
from model import build_model  # Assuming your model definition is in model.py
from vocabulary import Vocabulary  # Assuming vocabulary handling

# --- Replace with your actual configuration and paths ---
config_path = "Configs/Base.yaml"  # Path to your model configuration file
model_path = "Models/Base/checkpoint_best.pth.tar"  # Path to your trained model checkpoint
src_vocab_path = "Configs/src_vocab.txt"  # Path to your source vocabulary file

# Load configuration (you might need to adapt this based on how your config is loaded)
# Assuming a simple YAML loading mechanism:
import yaml
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Load vocabularies (adapt this to your actual vocabulary loading)
src_vocab = Vocabulary(file=src_vocab_path)
trg_vocab = [None] * (cfg['model']['trg_size'] + 1)  # Placeholder, adjust as needed

# Build the model (ensure this matches your model creation in training)
model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)

# Load the trained model weights
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# --- Create sample input (replace with your actual input) ---
# Example: Assuming your model takes source text indices and target input
batch_size = 1
max_length = 10  # Example sequence length
src = torch.randint(0, len(src_vocab), (batch_size, max_length))  # Example source indices
trg_input = torch.randn(batch_size, max_length, cfg['model']['trg_size'] + 1)  # Example target input
src_lengths = torch.tensor([max_length] * batch_size)  # Example source lengths

# --- Export to ONNX ---
onnx_file_path = "sign_language_model.onnx"
torch.onnx.export(model,
                  (src, trg_input, None, src_lengths, None),  # Model inputs as a tuple
                  onnx_file_path,
                  input_names=['src', 'trg_input', 'src_mask', 'src_lengths', 'trg_mask'],
                  output_names=['skel_out', 'gloss_out'],  # Adjust output names if needed
                  dynamic_axes={'src': {1: 'sequence_length'},
                                'trg_input': {1: 'sequence_length'},
                                #'skel_out': {1: 'sequence_length'}, #May be needed depending on the model
                                },
                  verbose=True)

print(f"Model exported to ONNX: {onnx_file_path}")