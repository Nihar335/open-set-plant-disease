# LeafScan on Hugging Face Spaces (CPU).
# The app comes from one pinned commit of the GitHub repository. To update the Space, change COMMIT.
FROM python:3.12-slim

ENV COMMIT=b440bdc2710d2ba65ac5c59f107d40bc9469e039

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt fetch_app.py ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python fetch_app.py "$COMMIT" . && rm fetch_app.py && chown -R user:user .
USER user

ENV LEAFSCAN_PUBLIC=1 HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "-m", "stage11.server"]
