from faster_whisper import WhisperModel
from typing import List, Union, Any, Optional
import os
import re
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)


def initialize_whisper(
    model_size: str, device: str = "cuda", compute_type: str = "float32"
) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def extract_number(filename: str) -> int:
    match = re.search(r"audio-(\d+)\.wav", filename)
    if match:
        return int(match.group(1))
    return -1


def concat_previous(str_list: Optional[List[str]]) -> str:
    result = ""
    if str_list is None:
        return result

    for s in str_list:
        clean_str = s.strip()
        if result and not result.endswith((" ", ".", ",")):
            result += " "
        result += clean_str
    return result


def process_record(
    model: Union[WhisperModel, Any], d: str, out: str, numSents: int = 3
) -> None:
    previous: Optional[List[str]] = None
    files = sorted(os.listdir(d), key=extract_number)
    logging.info(f"Start looking for wavs in {d}... found {len(files)}")
    target_file_all = os.path.join(d, out, f"audio-all-prev{numSents}.txt")
    os.makedirs(os.path.dirname(target_file_all), exist_ok=True)
    with open(target_file_all, "w") as in_all_file:
        pass

    for file in files:
        pattern = r"audio-(\d+)\.wav"
        match = re.match(pattern, file)
        if match:
            data = os.path.join(d, file)
            target_file = os.path.join(
                d, out, file.replace(".wav", f"-prev{numSents}.txt")
            )
            logging.info(f"Storing to: {target_file}")
            prompt = (
                concat_previous(previous)
                if previous is not None and numSents != 0
                else None
            )
            logging.info(f"> File: {file} \prompt {prompt} \n[ {previous} ]")
            segments, info = model.transcribe(data, beam_size=5, initial_prompt=prompt)
            with open(target_file, "w") as file:
                pass
            with open(target_file_all, "a") as in_all_file:
                with open(target_file, "a") as in_file:
                    concat = ""
                    for segment in segments:
                        concat += segment.text

                    if not isinstance(previous, list):
                        previous = [concat]
                    else:
                        if len(previous) == numSents:
                            previous = previous[1:] + [concat]
                        else:
                            previous.append(concat)
                    print(concat, file=in_file)
                    print(concat.strip(), file=in_all_file)

        else:
            logging.debug(f"No matched by regex {pattern} - ({file})")


def process_all(
    input_prefix: str,
    out: str,
    whisper: Union[WhisperModel, Any],
    numSents: int = 3,
) -> None:
    for folder in os.listdir(input_prefix):
        if os.path.isdir(input_prefix + folder):
            for subfolder in os.listdir(input_prefix + folder):
                if os.path.isdir(input_prefix + folder + "/" + subfolder):
                    for file in os.listdir(input_prefix + folder + "/" + subfolder):
                        if file == "en.IStde.man.vert+ts":
                            input_file_d = input_prefix + folder + "/" + subfolder
                            process_record(whisper, input_file_d, out, numSents)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper model."
    )
    parser.add_argument("input_prefix", type=str, help="Path prefix to the input files")
    parser.add_argument("out", type=str, help="Output dir for results")
    parser.add_argument(
        "--model_size", type=str, default="large-v2", help="Size of the Whisper model"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for computation"
    )
    parser.add_argument(
        "--compute_type", type=str, default="float32", help="Compute type for the model"
    )
    parser.add_argument(
        "--num_sents",
        type=int,
        default=3,
        help="Number of sentences to consider for previous prompt",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all records instead of a single record",
    )

    return parser.parse_args()


args = parse_args()

logging.info("Start initializing whisper...")
whisper = initialize_whisper(args.model_size, args.device, args.compute_type)
logging.info("Whisper initialized")

logging.info(f"Output file set to be {args.out}")

path = args.input_prefix
if args.all:
    logging.info(f"Start processing add records on path {path}")
    process_all(path, args.out, whisper, args.num_sents)
    logging.info(f"Job done! (process_all)")
else:
    logging.info(f"Start processing record on path {path}")
    process_record(whisper, path, args.out, args.num_sents)
    logging.info(f"Job done! (process_record)")
