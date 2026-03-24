# built-in
import os
import sys
import itertools
from pathlib import Path

# 3rd-party
import cv2
import numpy as np

# internal
import cli

def get_images(folder: Path):
    if not folder.is_dir():
        return None

    def is_image(path: Path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp', '.gif')
        return path.is_file() and path.suffix() in image_extensions

    images = filter(is_image, folder.glob('*'))
    return list(sorted(images))


def apply_rotation(img: cv2.Mat, angle: float):
    if angle == 0:
        return img

    w, h = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))


# TODO
def apply_border(img, border_size, border_color):
    if border_size <= 0 or border_color is None:
        return img
    result = img.copy()
    h, w = result.shape[:2]
    border = min(border_size, h//4, w//4)
    result[0:border, :] = border_color
    result[h-border:h, :] = border_color
    result[:, 0:border] = border_color
    result[:, w-border:w] = border_color
    return result


def main():
    if len(sys.argv) == 3:
        folder_path = sys.argv[1]
        output_folder = sys.argv[2]
    elif len(sys.argv) == 2:
        folder_path = sys.argv[1]
        output_folder = None
    elif HAS_TKINTER:
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select input folder with card images")
        if folder_path:
            output_folder = filedialog.askdirectory(title="Select output folder")
        root.destroy()
        if not folder_path:
            print("No folder selected.")
            return
    else:
        # print("Usage: python card_reviewer_v2.2.py <input_folder> <output_folder>")
        # return
        folder_path = "in"
        output_folder = "out"
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' not a valid folder")
        return
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = get_card_images(folder_path)
    if not images:
        print(f"No images in '{folder_path}'")
        return
    
    print(f"\nCard Reviewer v2.2")
    print(f"="*50)
    print(f"Input: {folder_path}")
    if output_folder:
        print(f"Output: {output_folder}")
    print(f"Found {len(images)} images")
    print(f"\nControls:")
    print(f"  Q/E: Rotate 0.25deg left/right (fine)")
    print(f"  A/D: Rotate 1deg left/right")
    print(f"  Z/X: Rotate 90deg")
    print(f"  I/J/K/L: Move(10px) | 8/4/6/2: Move(1px)")
    print(f"  C: Click to pick border color")
    print(f"  B: Toggle border on/off")
    print(f"  +/-: Adjust border size")
    print(f"  Enter: Save and next")
    print(f"  Escape: Skip (no save)")
    print(f"  O: Quit")
    print(f"="*50)
    
    current_index = 0
    saved_count = 0
    skipped_count = 0
    
    align_x = 0
    align_y = 0
    
    while current_index < len(images):
        filename = images[current_index]
        filepath = os.path.join(folder_path, filename)
        
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error: Could not load {filename}")
            current_index += 1
            continue
        
        h, w = img.shape[:2]
        
        rotation = 0
        border_size = 30
        border_enabled = True
        border_color = (0, 0, 0)
        
        max_w, max_h = 800, 1000
        scale = min(max_w / w, max_h / h, 1.0)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        window_name = f"Card Reviewer - {current_index + 1}/{len(images)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, scaled_w, scaled_h + 80)
        
        picking_color = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal picking_color, border_color
            if event == cv2.EVENT_LBUTTONDOWN and picking_color:
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                if 0 <= orig_y < h and 0 <= orig_x < w:
                    color = img[orig_y, orig_x]
                    border_color = tuple(int(c) for c in color)
                    print(f"  Picked color: B={color[0]}, G={color[1]}, R={color[2]}")
                    picking_color = False
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            preview = img.copy()
            preview = apply_rotation(preview, rotation, w, h)
            if border_enabled and border_size > 0:
                preview = apply_border(preview, border_size, border_color)
            
            preview_scaled = cv2.resize(preview, (scaled_w, scaled_h))
            
            overlay = np.zeros((scaled_h + 80, scaled_w, 3), dtype=np.uint8)
            overlay[:] = (30, 30, 30)
            overlay[:scaled_h, :] = preview_scaled
            
            center_x = scaled_w // 2 + align_x
            center_y = scaled_h // 2 + align_y
            cross_size = 1000
            
            cv2.line(overlay, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 255, 0), 1)
            cv2.line(overlay, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 255, 0), 1)
            
            if border_enabled and border_color is not None:
                cv2.rectangle(overlay, (scaled_w - 60, 10), (scaled_w - 10, 40), border_color, -1)
                cv2.rectangle(overlay, (scaled_w - 60, 10), (scaled_w - 10, 40), (255, 255, 255), 1)
                cv2.putText(overlay, "COLOR", (scaled_w - 58, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            cv2.putText(overlay, f"{filename[:40]}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            status = f"Rot:{rotation} | Border:{border_size}px | {'ON' if border_enabled else 'OFF'} | Offset:({align_x},{align_y})"
            cv2.putText(overlay, status, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            cv2.putText(overlay, "Q/E:0.25deg | A/D:1deg | Z/X:90deg | I/J/K/L:Move(10px) | 8/4/6/2:Move(1px) | C:Pick | B:Toggle | +/-:Size | ENTER:Save | ESC:Skip | O:Quit", 
                        (10, scaled_h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            
            cv2.imshow(window_name, overlay)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('o') or key == ord('O'):
                cv2.destroyAllWindows()
                print("\nQuitting...")
                return
            
            elif key == 13:
                final_img = img.copy()
                final_img = apply_rotation(final_img, rotation, w, h)
                if border_enabled and border_size > 0:
                    final_img = apply_border(final_img, border_size, border_color)
                
                if output_folder:
                    out_path = os.path.join(output_folder, filename)
                    cv2.imwrite(out_path, final_img)
                    print(f"  ✓ Saved to {output_folder}: {filename}")
                else:
                    cv2.imwrite(filepath, final_img)
                    print(f"  ✓ Saved: {filename}")
                
                saved_count += 1
                current_index += 1
                break
            
            elif key == 27:
                print(f"  - Skipped: {filename}")
                skipped_count += 1
                current_index += 1
                break
            
            elif key == ord('i') or key == ord('I'):
                align_y -= 10
            
            elif key == ord('k') or key == ord('K'):
                align_y += 10
            
            elif key == ord('j') or key == ord('J'):
                align_x -= 10
            
            elif key == ord('l') or key == ord('L'):
                align_x += 10
            
            elif key == 56:
                align_y -= 1
            
            elif key == 50:
                align_y += 1
            
            elif key == 52:
                align_x -= 1
            
            elif key == 54:
                align_x += 1
            
            elif key == ord('z') or key == ord('Z'):
                rotation = (rotation - 90) % 360
            
            elif key == ord('x') or key == ord('X'):
                rotation = (rotation + 90) % 360
            
            elif key == ord('q') or key == ord('Q'):
                rotation -= 0.25
            
            elif key == ord('e') or key == ord('E'):
                rotation += 0.25
            
            elif key == ord('a') or key == ord('A'):
                rotation -= 1
            
            elif key == ord('d') or key == ord('D'):
                rotation += 1
            
            elif key == ord('c') or key == ord('C'):
                print("  Click on border color...")
                picking_color = True
            
            elif key == ord('b') or key == ord('B'):
                border_enabled = not border_enabled
                print(f"  Border: {'ON' if border_enabled else 'OFF'}")
            
            elif key == ord('+') or key == ord('='):
                border_size += 1
            
            elif key == ord('-') or key == ord('_'):
                border_size = max(0, border_size - 1)
        
        cv2.destroyWindow(window_name)
    
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Saved: {saved_count}")
    print(f"  Skipped: {skipped_count}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
