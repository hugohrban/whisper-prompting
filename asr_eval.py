#!/bin/bash

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyctcdecode import build_ctcdecoder
import librosa

from functools import lru_cache
import sys


class ASRProcessor:
    def load_proc_model(self, proc_cls, model_cls, kenlm, alpha_beta=(None, None)):
        print(f"{self.model_id} is loading", file=sys.stderr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("using device: ", self.device, file=sys.stderr)

        self.asr_processor = proc_cls.from_pretrained(
            self.model_id, cache_dir="cache-dir"
        )
        self.asr_model = model_cls.from_pretrained(
            self.model_id, cache_dir="cache-dir"
        ).to(self.device)

        vock = list(self.asr_processor.tokenizer.get_vocab().keys())
        self.vock = vock

        alpha, beta = alpha_beta
        self.decoder = build_ctcdecoder(vock, kenlm, alpha=alpha, beta=beta)

    def asr(self, arr):
        input_values = self.asr_processor(
            arr, return_tensors="pt", sampling_rate=16000
        ).input_values  # Batch size 1
        logits = (
            self.asr_model(input_values.to(self.device))
            .logits.cpu()
            .detach()
            .numpy()[0]
        )
        l = self.decoder.decode(logits)
        return l


class Wav2vecASRProcessor(ASRProcessor):
    # this one also works
    # model_id = "facebook/wav2vec2-base-960h"

    def __init__(self, lan, model_id=None, lm=None, alpha_beta=(None, None)):
        if model_id is None:
            if lan == "en":
                model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

            elif lan == "de":
                model_id = "jonatasgrosman/wav2vec2-xls-r-1b-german"

        if lm is not None:
            kenlm = lan + "_5gram_lm.bin"
        else:
            kenlm = None

        self.model_id = model_id

        self.load_proc_model(
            Wav2Vec2Processor, Wav2Vec2ForCTC, kenlm, alpha_beta=alpha_beta
        )


class WhisperASRProcessor(ASRProcessor):
    def __init__(self, lan, model_id=None):
        if model_id is None:
            self.model_id = "openai/whisper-medium"
        else:
            self.model_id = model_id
        self.load_proc_model(
            WhisperProcessor, WhisperForConditionalGeneration, kenlm=None
        )
        self.asr_model.config.forced_decoder_ids = (
            self.asr_processor.get_decoder_prompt_ids(language=lan, task="transcribe")
        )

    def asr(self, arr):
        input_features = self.asr_processor(
            arr, sampling_rate=16000, return_tensors="pt"
        ).input_features
        predicted_ids = self.asr_model.generate(input_features.to(self.device))
        transcription = self.asr_processor.batch_decode(predicted_ids)

        return transcription


@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a


from cache_to_disk import cache_to_disk


@cache_to_disk(100)
def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def process_stream(proc, stream):
    for line in stream:
        p, beg, end, *ref = line.split("\t")
        a = load_audio_chunk(p, float(beg), float(end))

        dec = proc.asr(a)
        print(dec, flush=True)
        print("dec:", dec, file=sys.stderr)
        print("ref:", ref, file=sys.stderr)
        print(file=sys.stderr)


def main():
    from argparse import ArgumentParser

    ap = ArgumentParser(description="Eval ASR")
    ap.add_argument(
        "--lan", type=str, help="Source language, either en or de.", default="en"
    )
    ap.add_argument("--model", type=str, help="Either wav2vec or whisper.")
    ap.add_argument(
        "--model_id",
        type=str,
        help="Use this model ID. This option implies that --model type is correct.",
    )
    ap.add_argument(
        "--lm",
        action="store_true",
        help="Use language model {lan}_5gram_lm.bin for wav2vac model.",
        default=None,
    )
    # TODO: --lm_path

    ap.add_argument(
        "-a", help="alpha parameter of pyctcdecoding", type=float, default=None
    )
    ap.add_argument(
        "-b", help="beta parameter of pyctcdecoding", type=float, default=None
    )

    args = ap.parse_args()

    lm = args.lm

    lan = args.lan

    a, b = args.a, args.b
    if lan not in ["en", "de"]:
        print("wrong language option, only en and de are available", file=sys.stderr)
        sys.exit(1)

    if args.model == "whisper":
        print(f"whisper model will be used with lan = {a}", file=sys.stderr)
        proc = WhisperASRProcessor(lan, model_id=args.model_id)
    elif args.model == "wav2vec":
        print(f"Wav2vec model will be used with lan = {a}", file=sys.stderr)
        proc = Wav2vecASRProcessor(
            lan, model_id=args.model_id, lm=lm, alpha_beta=(a, b)
        )
    else:
        print("wrong model option, exiting", file=sys.stderr)
        sys.exit(1)

    process_stream(proc, sys.stdin)


if __name__ == "__main__":
    main()
