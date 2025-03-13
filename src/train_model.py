import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

# Training function
def train_model(model, dataset, device, num_epochs, batch_size, learning_rate=0.0003):
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,betas=(0.9, 0.95))

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for src, tgt in progress_bar:
            src = torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in src], padding_value=0, batch_first=True).to(device)
            tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tgt], padding_value=0, batch_first=True).to(device)

            # Shift the target sequence: decoder input and target output
            decoder_input = tgt[:, :-1]   # All tokens except the last one
            target_output = tgt[:, 1:]      # All tokens except the first one

            optimizer.zero_grad()
            output = model(src, decoder_input)
            loss = criterion(output.reshape(-1, model.tgt_vocab_size), target_output.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")
