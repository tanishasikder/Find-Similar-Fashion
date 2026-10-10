import colorsys
from PIL import Image

# Fixed order so the color head always uses the same index for each color
COLOR_NAMES = [
    'black', 'white', 'gray', 'brown', 'red', 'orange',
    'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
]

def get_colors():
    # Pytorch datasets encode labels with data
    return {name: i for i, name in enumerate(COLOR_NAMES)}

def dominant_rgb(img, num_colors=4):
    '''
    Gets the most common color in the middle of the crop. The edges of
    a bounding box are usually background so only the center is used.
    '''
    img = img.convert('RGB')
    w, h = img.size
    center = img.crop((w // 4, h // 4, w - w // 4, h - h // 4))
    if center.width == 0 or center.height == 0:
        center = img # Crop is too small to take the center of

    center.thumbnail((64, 64))
    # Quantize groups similar shades together then take the biggest group
    quantized = center.quantize(colors=num_colors)
    palette = quantized.getpalette()
    _, index = max(quantized.getcolors())
    return tuple(palette[index * 3:index * 3 + 3])

def rgb_to_hex(img):
    # Convert img to rgb value then return hex format
    return '#{:02x}{:02x}{:02x}'.format(*dominant_rgb(img))

def hex_to_category(img):
    '''
    Fashionpedia does not have color. Map images with
    hexcodes then define a category with this.
    '''
    hex = rgb_to_hex(img)
    hex_code = hex.lstrip('#')
    r, g, b = (int(hex_code[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360

    # achromatic cases first
    if v < 0.15:
        return 'black'
    if s < 0.10:
        return 'white' if v > 0.85 else 'gray'

    # brown: dark-ish orange/red/yellow hues
    if 15 <= h < 50 and v < 0.6:
        return 'brown'

    # hue bins
    bins = [
        (15, 'red'), (45, 'orange'), (70, 'yellow'),
        (165, 'green'), (200, 'cyan'), (260, 'blue'),
        (300, 'purple'), (345, 'pink'), (360, 'red'),
    ]
    for upper, name in bins:
        if h < upper:
            return name
    return 'red'
