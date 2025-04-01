FROM python:3.10-slim

WORKDIR /app

# Copy the dependency file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Run the main application (adjust the command as needed)
CMD ["python", "src/llm/main.py"]