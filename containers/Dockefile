# Use the base image from Swarmourr's federated learning container
FROM swarmourr/federated_learning_container_sub:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running (optional, if you don't want it to exit immediately)
#CMD ["tail", "-f", "/dev/null"]
