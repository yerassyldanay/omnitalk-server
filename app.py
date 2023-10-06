import argparse
import importlib.metadata
import queue
import time
import json
from typing import Callable
import numpy as np
import sounddevice as sd
import pywhispercpp.constants as constants
import webrtcvad
import logging
from pywhispercpp._logger import set_log_level
from pywhispercpp.model import Model
from llama_cpp import Llama
import os
import datetime
import subprocess
import platform
import requests

__version__ = importlib.metadata.version('pywhispercpp')

__header__ = f"""
=====================================
PyWhisperCpp + Llama Integration
Version: {__version__}               
=====================================
"""


class Assistant:
    """
    Assistant class

    Example usage
    ```python
    from pywhispercpp.examples.assistant import Assistant

    my_assistant = Assistant(commands_callback=print, n_threads=8)
    my_assistant.start()
    ```
    """

    def __init__(self, model='base', input_device: int = None, silence_threshold: int = 8, 
                q_threshold: int = 16, block_duration: int = 30, commands_callback: Callable[[str], None] = None, 
                model_log_level: int = logging.INFO, prompt="", **model_params):

        """
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param input_device: The input device (aka microphone), keep it None to take the default
        :param silence_threshold: The duration of silence after which the inference will be running
        :param q_threshold: The inference won't be running until the data queue is having at least `q_threshold` elements
        :param block_duration: minimum time audio updates in ms
        :param commands_callback: The callback to run when a command is received
        :param model_log_level: Logging level
        :param model_params: any other parameter to pass to the whsiper.cpp model see ::: pywhispercpp.constants.PARAMS_SCHEMA
        """

        initial_prompt_value = model_params.get('initial_prompt', None)  # Extract the custom value or use None as default
        threads = model_params.get('n_threads', 8)

        self.prompt = prompt

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # same as whisper.cpp
        self.channels = 1  # same as whisper.cpp
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()

        self.vad = webrtcvad.Vad(2)
        self.silence_threshold = silence_threshold
        self.q_threshold = q_threshold
        self._silence_counter = 0

        self.pwccp_model = Model(model,
                                 log_level=model_log_level,
                                 print_realtime=False,
                                 print_progress=False,
                                 print_timestamps=False,
                                 single_segment=True,
                                 no_context=True,
                                 initial_prompt=initial_prompt_value,
                                #  translate=True,
                                #  language='fr',
                                #  speed_up=True,
                                 n_threads=8)
        self.commands_callback = commands_callback

    def _audio_callback(self, indata, frames, time, status):
        """
        This is called (from a separate thread) for each audio block.
        """
        if status:
            logging.warning(F"underlying audio stack warning:{status}")

        assert frames == self.block_size
        audio_data = map(lambda x: (x + 1) / 2, indata)  # normalize from [-1,+1] to [0,1]
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        detection = self.vad.is_speech(audio_data, self.sample_rate)
        if detection:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    self._transcribe_speech()
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self):
        logging.info(f"Speech detected ...")
        audio_data = np.array([])
        while self.q.qsize() > 0:
            # get all the data from the q
            audio_data = np.append(audio_data, self.q.get())
        # Appending zeros to the audio data as a workaround for small audio packets (small commands)
        audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])
        # running the inference
        self.pwccp_model.transcribe(audio_data,
                                    new_segment_callback=self._new_segment_callback)

    def _new_segment_callback(self, seg):
        input_text = self.prompt + seg[0].text if self.prompt else seg[0].text
        print(input_text)

        # Check for specific commands
        if "open google" in seg[0].text.lower():
            subprocess.call(["open", "-a", "Google Chrome"])
            print("Opening Google Chrome...")
            return
        elif "search for" in seg[0].text.lower():
            query = seg[0].text.lower().replace("search for", "").strip()
            url = f"https://www.google.com/search?q={query}"
            subprocess.call(["open", url])
            return
        elif "open" in seg[0].text.lower() and "word" in seg[0].text.lower():
            subprocess.call(["open", "-a", "Microsoft Word"])
            return
        elif "show me the video" in seg[0].text.lower():
            video_path = "/Users/abdul/Downloads/IMG_4597.mov"
            self._play_video(video_path)
            return

        # New API call section
        api_url = "http://localhost:8000/v1/completions"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": input_text,
            "max_tokens": 200,
            "stop": ["Morgan:"],
            "echo": False,
            "stream": True
        }

        with requests.post(api_url, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                print("Error with the API call.")
                return
            accumulated_text = ""

            for line in response.iter_lines():
                if line:  # filter out keep-alive newlines
                    # print(line)

                    # Check if line starts with 'data: ' (after decoding from bytes)
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_data = decoded_line[len('data: '):]
                        try:
                            chunk_data = json.loads(json_data)
                            response_text = chunk_data.get('choices', [{}])[0].get('text', '')
                            # print(response_text)

                            # Accumulate text
                            accumulated_text += response_text
                            # print(accumulated_text)
                            accumulated_text = accumulated_text.replace("AIDEN:", "", 1)

                            # Check for end-of-sentence punctuation
                            if any(punctuation in response_text for punctuation in ['.', '!', '?']):
                                # Use the say command to speak the accumulated sentence
                                if platform.system() == "Darwin":  # macOS
                                    print(accumulated_text)
                                    subprocess.run(["say", accumulated_text])
                                elif platform.system() == "Linux":
                                    os.system(f"espeak '{accumulated_text}'")

                                # Clear accumulated_text for the next sentence
                                accumulated_text = ""

                        except json.JSONDecodeError as e:
                            print(f"Failed to decode JSON: {e}")

        if self.commands_callback:
            self.commands_callback(seg[0].text)

    def _play_video(self, video_path):
        subprocess.call(["open", video_path])

    def start(self) -> None:
        """
        Use this function to start the assistant
        :return: None
        """
        logging.info(f"Starting Assistant ...")
        with sd.InputStream(
                device=self.input_device,  # the default input device
                channels=self.channels,
                samplerate=constants.WHISPER_SAMPLE_RATE,
                blocksize=self.block_size,
                callback=self._audio_callback):

            try:
                logging.info(f"Assistant is listening ... (CTRL+C to stop)")
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logging.info("Assistant stopped")

    @staticmethod
    def available_devices():
        return sd.query_devices()


def _main():
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Args for Assistant
    parser.add_argument('-w', '--whisper_model', default='tiny.en', type=str, help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-ind', '--input_device', type=int, default=None, help=f'Input device (microphone)')
    parser.add_argument('-st', '--silence_threshold', default=16, type=int, help="Duration of silence for inference")
    parser.add_argument('-bd', '--block_duration', default=30, type=int, help="Minimum time audio updates in ms")
    # Args for Llama
    # parser.add_argument("-m", "--model", type=str, default="../llama.cpp/models/llama-2-7b-chat.ggmlv3.q8_0.bin", help="Llama model path")
    parser.add_argument("-pf", "--prompt-file", type=str, default="", help="Path to a text file that contains the prompt to guide the conversation direction with Llama.")
    parser.add_argument("-p", "--person", type=str, default="Morgan", help="Name of the person the bot is speaking to.")
    args = parser.parse_args()

    # Extract the prompt from the file
    prompt_content = ""
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as file:
            prompt_content = file.read().strip()
    
    # Capture the current time and year
    current_time = datetime.datetime.now().strftime('%H:%M')
    current_year = datetime.datetime.now().strftime('%Y')
    bot_name = "AIDEN"
    chat_symb = ":"
    my_prompt = f"A conversation with a person called {bot_name}."

    # Replace placeholders in the prompt
    prompt_content = prompt_content.format(args.person, bot_name, current_time, current_year, chat_symb)
    # print(prompt_content)

    # Initialize and start the Assistant
    my_assistant = Assistant(
                             model=args.whisper_model,
                             input_device=args.input_device,
                             silence_threshold=args.silence_threshold,
                             block_duration=args.block_duration,
                             commands_callback=print,
                             prompt=prompt_content,
                             model_params={'initial_prompt': my_prompt, 'n_threads': 8})
    my_assistant.start()


if __name__ == '__main__':
    _main()


# python assistantv2.py --model models/llama-2-7b-chat.q4_0.gguf --whisper_model models/ggml-base.en.bin --prompt-file prompts/talk-aiden2.txt
