from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()


def save_images_side_by_side_with_logits(
    og_img, adv_img, og_logits, adv_logits, caption, save_path
):
    og_img = to_pil(og_img) if not isinstance(og_img, Image.Image) else og_img
    adv_img = to_pil(adv_img) if not isinstance(adv_img, Image.Image) else adv_img

    canvas_width, canvas_height = 1280, 720
    caption_height = 50
    logits_text_height = 40
    vertical_padding = 20

    usable_height = canvas_height - caption_height - logits_text_height - vertical_padding * 2

    max_img_width = (canvas_width // 2) - 30

    def resize_preserve_aspect(img):
        w, h = img.size
        scale = min(max_img_width / w, usable_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    og_img_resized = resize_preserve_aspect(og_img)
    adv_img_resized = resize_preserve_aspect(adv_img)

    result_img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(result_img)

    try:
        font_path = "arial.ttf"
        ImageFont.truetype(font_path, 20)
        caption_font = ImageFont.truetype(font_path, 28)
    except IOError:
        ImageFont.load_default()
        caption_font = ImageFont.load_default()

    caption_bbox = draw.textbbox((0, 0), caption, font=caption_font)
    caption_width = caption_bbox[2] - caption_bbox[0]
    caption_x = (canvas_width - caption_width) // 2
    caption_y = 10
    draw.text((caption_x, caption_y), caption, fill="black", font=caption_font)

    og_x = 15
    og_y = caption_height + vertical_padding + (usable_height - og_img_resized.height) // 2
    adv_x = canvas_width // 2 + 15
    adv_y = caption_height + vertical_padding + (usable_height - adv_img_resized.height) // 2

    result_img.paste(og_img_resized, (og_x, og_y))
    result_img.paste(adv_img_resized, (adv_x, adv_y))

    try:
        logits_font = ImageFont.truetype(font_path, 14)
    except Exception:
        logits_font = ImageFont.load_default()

    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def draw_logits_and_classes(x, logits, max_idx, y_start):
        x_cursor = x
        draw.text((x_cursor, y_start), "Logits:", fill="black", font=logits_font)
        x_cursor += 55
        for i, logit in enumerate(logits):
            logit_str = f"{logit:.2f}"
            color = "red" if i == max_idx else "black"
            draw.text((x_cursor, y_start), logit_str, fill=color, font=logits_font)
            x_cursor += logits_font.getlength(logit_str) + 8

        x_cursor = x
        y_class = y_start + 18
        draw.text((x_cursor, y_class), "Classes:", fill="black", font=logits_font)
        x_cursor += 55
        for i, cname in enumerate(cifar10_classes):
            color = "red" if i == max_idx else "black"
            draw.text((x_cursor, y_class), cname, fill=color, font=logits_font)
            x_cursor += logits_font.getlength(cname) + 8

    og_max_idx = int(max(range(len(og_logits)), key=lambda i: og_logits[i]))
    adv_max_idx = int(max(range(len(adv_logits)), key=lambda i: adv_logits[i]))

    logits_y = caption_height + vertical_padding + usable_height + 5
    draw_logits_and_classes(og_x, og_logits, og_max_idx, logits_y)
    draw_logits_and_classes(adv_x, adv_logits, adv_max_idx, logits_y)

    result_img.save(save_path)
    # print(f"Saved fixed size (1280x720) image to {save_path}")
