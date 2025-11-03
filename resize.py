from pathlib import Path
from PIL import Image
import sys


TARGET_SIZE = (200, 72)
CONTENT_SIZE = (200, 60)  # 리사이즈할 이미지 크기
TOP_PADDING = 12          # 상단 패딩

def resize_pngs(root: Path) -> tuple[int, int]:
	"""Recursively resize all PNG images under root to TARGET_SIZE.

	Resizes to 200x60px first, adds 12px top padding.
	Converts transparent background to white.
	Returns (processed_count, failed_count).
	"""
	processed = 0
	failed = 0

	if not root.exists():
		print(f"Target directory does not exist: {root}")
		return processed, 0

	for p in root.rglob("*"):
		if not p.is_file():
			continue
		if p.suffix.lower() != ".png":
			continue

		try:
			with Image.open(p) as img:
				orig_width, orig_height = img.size
				
				# Preserve alpha if present
				has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

				# Convert transparent background to white
				if has_alpha:
					# Create white background
					white_bg = Image.new("RGB", img.size, (255, 255, 255))
					# Convert image to RGBA if needed
					img = img.convert("RGBA")
					# Composite image over white background
					white_bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
					img = white_bg

				# Ensure RGB mode
				img = img.convert("RGB")

				# Resize to 200x60 (ignoring aspect ratio)
				resized = img.resize(CONTENT_SIZE, Image.LANCZOS)
				
				# Create new image with TARGET_SIZE and white background
				new_img = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
				
				# Paste resized image with top padding
				paste_x = 0
				paste_y = TOP_PADDING
				new_img.paste(resized, (paste_x, paste_y))

				# Save back to PNG
				new_img.save(p, format="PNG")
				processed += 1
				print(f"Resized: {p} ({orig_width}x{orig_height} -> {CONTENT_SIZE[0]}x{CONTENT_SIZE[1]} with {TOP_PADDING}px top padding)")
		except Exception as e:
			failed += 1
			print(f"Failed to process {p}: {e}")

	return processed, failed


def main():
	# Compute target dir relative to script location
	base = Path(__file__).resolve().parent
	target = base / "captcha_data" / "gov24" / "0" / "images"

	processed, failed = resize_pngs(target)

	print("---")
	print(f"Processed: {processed}")
	print(f"Failed: {failed}")


if __name__ == "__main__":
	main()
