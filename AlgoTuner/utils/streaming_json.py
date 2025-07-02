"""
utils/streaming_json.py

Provides a streaming JSONL loader using ijson for streaming and a json.loads fallback.
"""

import logging
import orjson
from typing import Iterator, Dict, Any, Optional
from AlgoTuner.utils.serialization import dataset_decoder
import os
import traceback
import functools # Import functools


def stream_jsonl(file_path: str, decoder_base_dir: Optional[str] = None) -> Iterator[Dict]:
    """
    Lazily parse a JSONL file one record at a time using orjson.loads.
    Passes base_dir for external reference decoding via functools.partial.
    """
    # Use provided decoder_base_dir if available, otherwise derive from file_path
    # This allows flexibility if the .npy files are not in the same dir as the .jsonl
    # (though current setup implies they are in a subdir of the .jsonl's dir)
    actual_base_dir = decoder_base_dir if decoder_base_dir is not None else os.path.dirname(file_path)

    logging.info(f"Streaming {file_path} using orjson.loads (with dataset_decoder for external refs, base_dir: {actual_base_dir})")
    try:
        with open(file_path, 'r') as f:
            # Create the object_hook using functools.partial to include actual_base_dir
            object_hook_for_load = functools.partial(dataset_decoder, base_dir=actual_base_dir)

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # Step 1: Parse JSON with orjson to a raw Python dictionary
                    raw_record = orjson.loads(line)
                    # Step 2: Apply the custom dataset_decoder to the raw dictionary
                    processed_record = object_hook_for_load(raw_record)
                    yield processed_record
                except orjson.JSONDecodeError as e: # Catch orjson specific decode error
                    logging.error(f"JSON Decode Error in {file_path}, line {line_num}: {e}")
                    # Decide how to handle: skip line, raise error, etc.
                    # For robustness, let's log and skip the line.
                    continue
                except Exception as e:
                    logging.warning(f"Skipping malformed JSON line in {file_path}, line {line_num}: {e}")
                    logging.warning(traceback.format_exc()) # Add traceback for unexpected errors
    except FileNotFoundError:
        logging.error(f"File not found during streaming: {file_path}")
        # Return an empty iterator if file not found
        return iter([])
    except Exception as e_open:
        logging.error(f"Error opening or reading file {file_path}: {e_open}")
        logging.error(traceback.format_exc())
        # Return an empty iterator on other file errors
        return iter([]) 