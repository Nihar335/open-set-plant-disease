# LeafScan on Hugging Face Spaces (CPU).
# The app comes from one pinned commit of the GitHub repository. To update the Space, change COMMIT.
FROM python:3.12-slim

ENV COMMIT=PIN_AFTER_PUSH

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt fetch_app.py ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python fetch_app.py "$COMMIT" . && rm fetch_app.py && chown -R user:user .
USER user

ENV LEAFSCAN_PUBLIC=1 HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "-m", "stage11.server"]
