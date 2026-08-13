# Running the web interface

## 1. Install Streamlit

```
pip install streamlit
```

## 2. Start the app

From the project root:

```
streamlit run app_streamlit.py --server.maxUploadSize 500 --server.fileWatcherType none
```

## 3. Open the page

The terminal prints an address, usually:

```
http://localhost:8501
```

Open it in a browser.

## 4. Generate a character

Upload a video on the left, check the preview is the right way up, and press
**Generate**. It takes a few minutes. The character appears on the right, and
the three-panel comparison below it.

## Stopping

Press `Ctrl+C` in the terminal.


