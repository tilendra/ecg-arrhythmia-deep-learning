# ===============================
# Base Image (Pinned)
# ===============================
FROM python:3.11.8-slim-bookworm

# ===============================
# System-Level Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential=12.9 \
    gcc=4:12.2.0-3 \
    g++=4:12.2.0-3 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Environment Settings
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# ===============================
# Install Python Dependencies
# ===============================
COPY requirements.txt .

RUN pip install --upgrade pip==24.0
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy App Files
# ===============================
COPY . .

# ===============================
# Expose Port
# ===============================
EXPOSE 8501

# ===============================
# Run Streamlit
# ===============================
CMD ["streamlit", "run", "ecg_arrhythmia_detection_app_final.py"]