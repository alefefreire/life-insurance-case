FROM python:3.9.16-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  curl \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock (if exists)
COPY pyproject.toml poetry.lock* ./
# Install project dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Install Jupyter and create kernel
RUN poetry run pip install jupyterlab \
  && poetry run python -m ipykernel install --user --name life-insurance --display-name "life-insurance Kernel"

# Expose Jupyter port
EXPOSE 8888

# Default command to start Jupyter Lab
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]