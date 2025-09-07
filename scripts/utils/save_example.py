from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

def save_images_side_by_side_with_logits(og_img, adv_img, og_logits, adv_logits, caption, save_path):
    og_img = to_pil(og_img) if not isinstance(og_img, Image.Image) else og_img
    adv_img = to_pil(adv_img) if not isinstance(adv_img, Image.Image) else adv_img

    # Final canvas size
    canvas_width, canvas_height = 1280, 720

    # Space reserved for caption and logits text
    caption_height = 50
    logits_text_height = 40
    vertical_padding = 20

    # Usable height area for images inside the canvas
    usable_height = canvas_height - caption_height - logits_text_height - vertical_padding * 2

    # Max width per image (split canvas width in half, less some padding)
    max_img_width = (canvas_width // 2) - 30

    # Resize each image to fit within max_img_width and usable_height, preserving aspect ratio
    def resize_preserve_aspect(img):
        w, h = img.size
        scale = min(max_img_width / w, usable_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    og_img_resized = resize_preserve_aspect(og_img)
    adv_img_resized = resize_preserve_aspect(adv_img)

    # Create blank white canvas
    result_img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(result_img)

    # Select font: replace with a valid ttf path or fall back to default
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 20)
        caption_font = ImageFont.truetype(font_path, 28)
    except IOError:
        font = ImageFont.load_default()
        caption_font = ImageFont.load_default()

    # Draw caption centered at top
    caption_bbox = draw.textbbox((0, 0), caption, font=caption_font)
    caption_width = caption_bbox[2] - caption_bbox[0]
    caption_x = (canvas_width - caption_width) // 2
    caption_y = 10
    draw.text((caption_x, caption_y), caption, fill="black", font=caption_font)

    # Calculate positions to paste images (centered vertically in usable space)
    og_x = 15
    og_y = caption_height + vertical_padding + (usable_height - og_img_resized.height) // 2
    adv_x = canvas_width // 2 + 15
    adv_y = caption_height + vertical_padding + (usable_height - adv_img_resized.height) // 2

    result_img.paste(og_img_resized, (og_x, og_y))
    result_img.paste(adv_img_resized, (adv_x, adv_y))

    # Prepare logits text
    og_logits_str = "Logits: " + ", ".join(f"{logit:.2f}" for logit in og_logits)
    adv_logits_str = "Logits: " + ", ".join(f"{logit:.2f}" for logit in adv_logits)

    # Draw logits below each image
    logits_y = caption_height + vertical_padding + usable_height + 5
    draw.text((og_x, logits_y), og_logits_str, fill="black", font=font)
    draw.text((adv_x, logits_y), adv_logits_str, fill="black", font=font)

    # Save output
    result_img.save(save_path)
    print(f"Saved fixed size (1280x720) image to {save_path}")

# Example usage:
if __name__ == "__main__":
    og_img = Image.open("path_to_original_image.png")
    adv_img = Image.open("path_to_adversarial_image.png")

    og_logits = [2.1, -1.3, 0.5]
    adv_logits = [1.9, -0.9, 0.7]

    save_images_side_by_side_fixed_size(
        og_img, adv_img, og_logits, adv_logits,
        "Original vs Adversarial",
        "output_1280x720.png"
    )
