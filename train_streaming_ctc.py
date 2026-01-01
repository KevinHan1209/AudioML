"""
CTC training script that streams LibriSpeech examples via Hugging Face datasets.

This mirrors the logic from practice.ipynb but avoids any local LibriSpeech copy by
reusing StreamingLibriSpeechDataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

os.environ.setdefault("HF_DATASETS_AUDIO_BACKEND", "soundfile")

from streaming_librispeech_dataset import StreamingLibriSpeechDataset


class CharTokenizerCTC:
    def __init__(self) -> None:
        alphabet = list("abcdefghijklmnopqrstuvwxyz' ")
        self.blank_id = 0
        self.id2ch = ["<blank>"] + alphabet
        self.ch2id = {ch: i for i, ch in enumerate(self.id2ch)}
        self.vocab_size = len(self.id2ch)

    def normalize(self, text: str) -> str:
        text = text.lower()
        out = []
        for ch in text:
            out.append(ch if ch in self.ch2id else " ")
        s = "".join(out)
        while "  " in s:
            s = s.replace("  ", " ")
        return s.strip()

    def encode(self, text: str) -> torch.Tensor:
        normalized = self.normalize(text)
        if not normalized:
            return torch.empty(0, dtype=torch.long)
        ids = [self.ch2id[ch] for ch in normalized]
        return torch.tensor(ids, dtype=torch.long)


class BiLSTMCTC(nn.Module):
    def __init__(self, n_mels: int, vocab_size: int, hidden: int = 256, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden * 2, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(features)
        logits = self.proj(h)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)


def make_ctc_targets(tokenizer: CharTokenizerCTC, batch_text: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode(t) for t in batch_text]
    target_lengths = torch.tensor([e.numel() for e in encoded], dtype=torch.long)
    non_empty = [e for e in encoded if e.numel() > 0]
    if non_empty:
        targets_1d = torch.cat(non_empty, dim=0)
    else:
        targets_1d = torch.empty(0, dtype=torch.long)
    return targets_1d, target_lengths


def collate_waveforms_ctc(batch):
    waveforms, logmels, sample_rates, transcripts = zip(*batch)
    logmels_tf = [lm.transpose(0, 1) for lm in logmels]
    input_lengths = torch.tensor([lm.shape[0] for lm in logmels_tf], dtype=torch.long)
    logmels_padded = pad_sequence(logmels_tf, batch_first=True, padding_value=0.0)
    return waveforms, logmels_padded, input_lengths, list(sample_rates), list(transcripts)


def build_dataloader(args) -> DataLoader:
    dataset = StreamingLibriSpeechDataset(
        subset=args.subset,
        split=args.split,
        sampling_rate=args.sample_rate,
        streaming=True,
        cache_dir=args.cache_dir,
        text_column="text",
        max_samples=args.max_samples,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_waveforms_ctc,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizerCTC()
    dataloader = build_dataloader(args)
    model = BiLSTMCTC(
        n_mels=args.n_mels,
        vocab_size=tokenizer.vocab_size,
        hidden=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        frames_seen = 0
        processed_steps = 0
        for step, (_, batch_logmels, input_lengths, _, batch_text) in enumerate(dataloader, start=1):
            batch_logmels = batch_logmels.to(device)
            input_lengths = input_lengths.to(device)
            targets_1d, target_lengths = make_ctc_targets(tokenizer, batch_text)
            if targets_1d.numel() == 0:
                continue
            targets_1d = targets_1d.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(batch_logmels)
            loss = criterion(log_probs, targets_1d, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * batch_logmels.size(0)
            frames_seen += input_lengths.sum().item()
            processed_steps += 1

            if processed_steps % args.log_interval == 0:
                avg_loss = running_loss / max(frames_seen, 1)
                print(f"Epoch {epoch} Step {processed_steps} - avg loss/frame: {avg_loss:.4f} - batch loss: {loss.item():.4f}")

            if args.steps_per_epoch and processed_steps >= args.steps_per_epoch:
                break

        if processed_steps == 0:
            print(f"Epoch {epoch} finished with no valid batches.")
        else:
            avg_loss = running_loss / max(frames_seen, 1)
            print(f"Epoch {epoch} complete. Processed steps={processed_steps}, avg loss/frame={avg_loss:.4f}")

    if args.checkpoint_path:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "tokenizer_state": {"id2ch": tokenizer.id2ch, "blank_id": tokenizer.blank_id},
            "config": {
                "n_mels": args.n_mels,
                "hidden": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
        }
        torch.save(ckpt, args.checkpoint_path)
        print(f"Saved checkpoint to {args.checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple BiLSTM-CTC model on streaming LibriSpeech.")
    parser.add_argument("--subset", default="clean", help="LibriSpeech config (clean/other/all).")
    parser.add_argument("--split", default="train.100", help="datasets split to stream.")
    parser.add_argument("--sample-rate", type=int, default=16_000, dest="sample_rate")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--pin-memory", action="store_true", dest="pin_memory")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--n-fft", type=int, default=400, dest="n_fft")
    parser.add_argument("--hop-length", type=int, default=160, dest="hop_length")
    parser.add_argument("--n-mels", type=int, default=80, dest="n_mels")
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=256, dest="hidden_size")
    parser.add_argument("--num-layers", type=int, default=3, dest="num_layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100, dest="steps_per_epoch")
    parser.add_argument("--log-interval", type=int, default=10, dest="log_interval")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("bilstm_ctc_checkpoint.pt"), dest="checkpoint_path")
    parser.add_argument("--cache-dir", type=Path, default=None, dest="cache_dir")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
