import colorsys
import sys

def rgb_to_hsv(r, g, b):
    # 将RGB值转换为HSV值
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 180, s * 255, v * 255

def main():
    if len(sys.argv) != 4:
        print("Usage: python rgb_to_hsv.py <R> <G> <B>")
        return
    
    try:
        r = int(sys.argv[1])
        g = int(sys.argv[2])
        b = int(sys.argv[3])
        
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values should be between 0 and 255")
        
        h, s, v = rgb_to_hsv(r, g, b)
        
        print(f"RGB({r}, {g}, {b}) -> HSV({h:.2f}, {s:.2f}, {v:.2f})")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("RGB values should be integers between 0 and 255")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
