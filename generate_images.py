from PIL import Image, ImageDraw, ImageFont
import os

# Create directory if it doesn't exist
os.makedirs("static/img", exist_ok=True)

# Function to create a simple pizza image
def create_pizza_image(filename, color, text):
    # Create a 400x300 image with the specified background color
    img = Image.new('RGB', (400, 300), color)
    d = ImageDraw.Draw(img)
    
    # Try to use a default font, or fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text to the image
    text_width, text_height = d.textbbox((0, 0), text, font=font)[2:4]
    text_x = (400 - text_width) // 2
    text_y = (300 - text_height) // 2
    d.text((text_x, text_y), text, fill="white", font=font)
    
    # Save the image
    img.save(f"static/img/{filename}")
    print(f"Created {filename}")

# Create images for each pizza
pizzas = [
    ("margherita.jpg", "#DE2910", "Margherita Pizza"),
    ("pepperoni.jpg", "#9C2F2F", "Pepperoni Pizza"),
    ("veggie.jpg", "#2F9C2F", "Veggie Pizza"),
    ("hawaiian.jpg", "#2F2F9C", "Hawaiian Pizza"),
    ("bbq_chicken.jpg", "#9C2F9C", "BBQ Chicken Pizza")
]

# Generate all images
for filename, color, text in pizzas:
    create_pizza_image(filename, color, text)

print("All pizza images have been generated.")