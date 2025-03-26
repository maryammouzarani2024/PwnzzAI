import requests
import os

# Create directory if it doesn't exist
os.makedirs("static/img", exist_ok=True)

# Pizza image URLs (public domain/freely usable images)
pizza_images = {
    "margherita.jpg": "https://cdn.pixabay.com/photo/2017/12/10/14/47/pizza-3010062_1280.jpg",
    "pepperoni.jpg": "https://cdn.pixabay.com/photo/2016/03/05/21/45/pizza-1239077_1280.jpg",
    "veggie.jpg": "https://cdn.pixabay.com/photo/2020/06/10/13/30/pizza-5282458_1280.jpg",
    "hawaiian.jpg": "https://cdn.pixabay.com/photo/2017/01/03/11/33/pizza-1949183_1280.jpg",
    "bbq_chicken.jpg": "https://cdn.pixabay.com/photo/2017/09/30/15/10/plate-2802332_1280.jpg"
}

# Download each image
for filename, url in pizza_images.items():
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(f"static/img/{filename}", 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("All images downloaded successfully!")