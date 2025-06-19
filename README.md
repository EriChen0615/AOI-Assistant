# Installtion

You need to install
* SQLite

```bash
pip install openai fastapi uvicorn
```

# Using AOI

Follow the following steps to install AOI:

1. Add your OpenAI API Key to configs/openai_api_key

2. Start the memory server with the following code

```bash
screen -S memoryserver
cd src/aoi-memory
uvicorn server:app --reload --port 5050 # this starts the SQLite Server
```

3. Start the chrome browser extension for ambient/active logging. Follow the steps below:
* Open chrome://extensions/, enable Developer Mode
* Load unpacked extension folder (i.e., `src/aoi-chrome-memory-tracker`)
* Visit any website
* Click your extension icon (short cut = Alt+A) → click “Start Memory Mode”
* Click a DOM element on the page for active logging


4. Start the main conversational agent

```bash
python demo/mini_aoi.py
```

There are a few global variables in `demo/mini_aoi.py` that you can tune.
* `INPUT_MODE`
* `OUTPUT_MODE`
* `LANGUAGE`

Enjoy!

Jinghong Chen. 2025.6
