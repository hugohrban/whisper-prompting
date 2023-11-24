from faster_whisper import WhisperModel
from typing import List, Union, Any, Optional, Dict, Tuple
import os
import re
import argparse
import logging
from jiwer import wer
from align_wer import open_transform, nice_alignments, process_asr_align


logging.basicConfig(level=logging.DEBUG)


def initialize_whisper(
    model_size: str, device: str = "cuda", compute_type: str = "float32"
) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def extract_number_eval_prev(filename: str) -> int:
    match = re.search(r"audio-(\d+)-prev(\d+)\.txt", filename)
    if match:
        return int(match.group(1))
    return -1


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


def calculate_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)


def evaluate_by_wer(d: str, out: str) -> None:
    files = sorted(os.listdir(os.path.join(d, out)), key=extract_number_eval_prev)
    print("files", files)
    gold_transcript = os.path.join(d, out, "audio-gold.txt")
    wer_file = os.path.join(d, out, "audio-wer.txt")
    audio_pattern = re.compile(r"audio-(\d+)-prev(\d+)\.txt")
    with open(gold_transcript, "r") as in_file:
        gold_text = in_file.read().strip().split("\n")

    groups: Dict[str, List[Tuple[str, float]]] = {}
    for file in files:
        match = audio_pattern.match(file)
        if match:
            # Extract the 'prev' number and the 'k' number
            k_number = match.group(1)
            prev_number = match.group(2)

            # Load the transcript of the current file
            # Assuming the transcript is stored in a similar-named text file
            transcript_path = os.path.join(d, f"audio-{k_number}-prev{prev_number}.txt")
            with open(transcript_path, "r") as transcript_file:
                transcript_text = transcript_file.read().strip()

            # Calculate WER
            wer_score = calculate_wer(gold_text[int(k_number) - 1], transcript_text)

            # Add the file and its WER score to the appropriate group
            if prev_number not in groups:
                groups[prev_number] = []
            groups[prev_number].append((file, wer_score))

    # Sort files in each group by k_number
    for prev in groups:
        groups[prev].sort(key=lambda x: extract_number(x[0]))

    # Print the groups and their WER scores
    open(wer_file, "w").close()
    # Writing to the WER file
    with open(wer_file, "w") as in_wer_file:
        for prev, file_data in groups.items():
            # Calculate WER for the audio-all-prev{i}.txt file
            all_transcript_path = os.path.join(d, out, f"audio-all-prev{prev}.txt")
            with open(all_transcript_path, "r") as all_transcript_file:
                all_transcript_text = all_transcript_file.read().strip()
            all_wer = calculate_wer("\n".join(gold_text), all_transcript_text)

            print(f"Previous sents {prev}:", file=in_wer_file)
            for file, score in file_data:
                print(f"  File: {file}, WER: {score:.2f}", file=in_wer_file)

                # Call the alignment script
                gold_fn = os.path.join(
                    d, out, f"audio-{extract_number_eval_prev(file)}-gold.txt"
                )
                asr_fn = os.path.join(d, out, file)
                tA, tB, ts = open_transform(gold_fn, asr_fn)
                na = nice_alignments(tA, tB)
                alignment_output = process_asr_align(tA, tB, na)

                # Print the alignment results
                for line in alignment_output:
                    print(line, file=in_wer_file)
            print(
                f"Total WER for audio-all-prev{prev}: {all_wer:.2f}\n", file=in_wer_file
            )


def process_record(
    model: Union[WhisperModel, Any], d: str, out: str, numSents: int = 3
) -> None:
    previous: Optional[List[str]] = None
    files = sorted(os.listdir(d), key=extract_number)
    logging.info(f"Start looking for wavs in {d}... found {len(files)}")
    target_file_all = os.path.join(d, out, f"audio-all-prev{numSents}.txt")
    gold_data_file = os.path.join(d, out, f"audio-gold.txt")
    os.makedirs(os.path.dirname(target_file_all), exist_ok=True)
    open(target_file_all, "w").close()

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
            open(target_file, "w").close()
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
        elif file == "en.OSt.man.orto.txt":
            with open(os.path.join(d, file), "r") as in_r_file:
                with open(gold_data_file, "w") as in_w_file:
                    print(in_r_file.read(), file=in_w_file)
        else:
            logging.debug(f"No matched by regex {pattern} - ({file})")

    evaluate_by_wer(d, out)


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


# logging.info("Start initializing whisper...")
# whisper = initialize_whisper(args.model_size, args.device, args.compute_type)
# logging.info("Whisper initialized")

logging.info(f"Output file set to be {args.out}")

path = args.input_prefix
if args.all:
    logging.info(f"Start processing add records on path {path}")
    process_all(path, args.out, whisper, args.num_sents)
    logging.info(f"Job done! (process_all)")
else:
    logging.info(f"Start processing record on path {path}")

    evaluate_by_wer(path, args.out)
    # process_record(whisper, path, args.out, args.num_sents)
    logging.info(f"Job done! (process_record)")
