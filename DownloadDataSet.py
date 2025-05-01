import kagglehub

# Download latest version
path = kagglehub.dataset_download("akash034/english-hindi-telugu-text-images")

print("Path to dataset files:", path)
