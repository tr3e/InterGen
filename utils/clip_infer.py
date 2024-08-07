import torch
import clip
import os
from datetime import datetime
import re

def sanitize_filename(text, default="output"):
    # Remove invalid characters for filenames and limit length
    sanitized_text = re.sub(r'[^a-zA-Z0-9]', '_', text)
    sanitized_text = sanitized_text[:50]  # Limit the length to 50 characters
    return sanitized_text if sanitized_text else default

def save_latent_text(raw_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)

    with torch.no_grad():
        text = clip.tokenize(raw_text, truncate=True).to(device)
        token_embedding = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
        positional_embedding = clip_model.positional_embedding.type(clip_model.dtype)
        x = token_embedding + positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        clip_out = clip_model.ln_final(x).type(clip_model.dtype)

    # Take the embedding of the [EOS] token (the highest index)
    latent_text = clip_out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    
    # Detach and convert to numpy
    latent_text_numpy = latent_text.detach().cpu().numpy()
    
    # Print the 1D tensor to the console
    print("1D Tensor for prompt '{}':".format(raw_text), latent_text_numpy)
    
    # Sanitize prompt text to create a valid filename
    filename = sanitize_filename(raw_text)
    
    # Create output directory if it doesn't exist
    os.makedirs("output/1D_tensor", exist_ok=True)
    file_path = os.path.join("output/1D_tensor", f"{filename}.txt")
    
    with open(file_path, "w") as file:
        file.write(str(latent_text_numpy))
    
    print(f"Latent text for prompt '{raw_text}' saved to {file_path}")

if __name__ == '__main__':
    with open("prompts.txt") as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]

    for text in texts:
        if text.strip():  # Ensure the prompt is not empty
            save_latent_text(text)
        else:
            print("Empty prompt detected, skipping.")
