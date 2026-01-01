"""
Streaming-friendly LibriSpeech dataset wrapper that keeps data off disk.

This module exposes StreamingLibriSpeechDataset, a torch IterableDataset that
pulls audio/text pairs from the Hugging Face datasets hub instead of a local
copy of LibriSpeech and lazily converts waveforms to log-mel spectrograms.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

from io import BytesIO

import fsspec
import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, IterableDataset as HFIterableDataset, load_dataset
from torch.utils.data import IterableDataset, get_worker_info


class StreamingLibriSpeechDataset(IterableDataset):
    """
    Torch IterableDataset that streams LibriSpeech via Hugging Face datasets.

    Examples
    --------
    >>> dataset = StreamingLibriSpeechDataset(split="train.100")
    >>> waveform, logmel, sr, text = next(iter(dataset))
    """

    def __init__(
        self,
        subset: str = "clean",
        split: str = "train.100",
        sampling_rate: int = 16_000,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        load_dataset_kwargs: Optional[Dict] = None,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> None:
        """
        Args:
            subset: LibriSpeech configuration ('clean', 'other', or 'all').
            split: Datasets split name such as 'train.100', 'validation.clean'.
            sampling_rate: Desired sampling rate for returned audio.
            streaming: Whether to enable datasets streaming mode.
            cache_dir: Optional cache directory for datasets artifacts.
            text_column: Column name containing transcripts.
            max_samples: Optional cap on yielded samples (useful for debugging).
            load_dataset_kwargs: Extra kwargs forwarded to load_dataset.
            n_fft: FFT size for MelSpectrogram.
            hop_length: Hop length for MelSpectrogram.
            n_mels: Number of mel filter banks.
            fmin: Minimum mel frequency.
            fmax: Maximum mel frequency. Defaults to sr/2 when None.
        """

        load_kwargs = load_dataset_kwargs.copy() if load_dataset_kwargs else {}
        self.base_dataset: HFIterableDataset = load_dataset(
            "librispeech_asr",
            subset,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            **load_kwargs,
        )
        self.dataset: HFIterableDataset = self.base_dataset.cast_column(
            "audio", Audio(sampling_rate=sampling_rate, decode=False)
        )
        self.sampling_rate = sampling_rate
        self.text_column = text_column
        self.max_samples = max_samples
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax or sampling_rate / 2,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int, str]]:
        worker = get_worker_info()
        if worker is None:
            iterable: Iterable = self.dataset
        else:
            iterable = self.dataset.shard(worker.num_workers, worker.id)

        yielded = 0
        for sample in iterable:
            audio_meta = sample["audio"]
            waveform, sample_rate = self.decode_audio(audio_meta)
            transcript = sample[self.text_column]
            waveform_tensor = torch.from_numpy(waveform).float()
            logmel_tensor = self.audio_to_melspec(waveform_tensor)
            yield waveform_tensor, logmel_tensor, sample_rate, transcript

            yielded += 1
            if self.max_samples is not None and yielded >= self.max_samples:
                break

    def decode_audio(self, audio_meta: Dict) -> Tuple["np.ndarray", int]:
        """
        Convert dataset audio metadata into waveform numpy array and sample rate.
        """

        path = audio_meta.get("path")
        data_bytes = audio_meta.get("bytes")
        stream = None
        try:
            if data_bytes is not None:
                stream = BytesIO(data_bytes)
            elif path is not None:
                stream = fsspec.open(path, "rb").open()
            else:
                raise ValueError("Audio sample missing both 'path' and 'bytes'.")
            waveform, sample_rate = sf.read(stream, dtype="float32", always_2d=False)
        finally:
            if stream is not None:
                stream.close()
        return waveform, sample_rate

    def audio_to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a waveform tensor (1D or [1, T]) to a log-mel spectrogram tensor.
        """

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_transform(waveform)
        logmel = self.db_transform(mel).squeeze(0)
        return logmel
