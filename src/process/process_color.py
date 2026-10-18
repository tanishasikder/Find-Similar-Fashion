import colorsys
import numpy as np
from PIL import Image

# Fixed order so the color head always uses the same index for each color
COLOR_NAMES = [
    'black', 'white', 'gray', 'brown', 'red', 'orange',
    'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
]

def get_colors():
    # Pytorch datasets encode labels with data
    return {name: i for i, name in enumerate(COLOR_NAMES)}

def skin_mask(pixels):
    '''
    True for pixels that look like skin. Uses the common YCbCr skin range,
    which works across skin tones since it ignores brightness (Y)
    '''
    r, g, b = (pixels[:, i].astype(np.float32) for i in range(3))
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return (cb >= 77) & (cb <= 127) & (cr >= 138) & (cr <= 173)

def dominant_rgb(img, num_colors=4):
    '''
    Gets the most common color in the middle of the crop. The edges of
    a bounding box are usually background so only the center is used.
    Skin is skipped so necklines and sleeveless tops get the garment color
    '''
    img = img.convert('RGB')
    w, h = img.size
    center = img.crop((w // 4, h // 4, w - w // 4, h - h // 4))
    if center.width == 0 or center.height == 0:
        center = img # Crop is too small to take the center of

    center.thumbnail((64, 64))
    pixels = np.asarray(center).reshape(-1, 3)
    # Drop skin, unless almost everything is skin-colored (tan/beige clothes
    # can fall in the skin range too, and then the garment is the skin-colored part)
    keep = pixels[~skin_mask(pixels)]
    if len(keep) >= 0.2 * len(pixels):
        pixels = keep

    # Quantize groups similar shades together then take the biggest group
    strip = Image.fromarray(pixels.reshape(1, -1, 3).astype(np.uint8))
    quantized = strip.quantize(colors=num_colors)
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
    Photos tint every neutral a little (lighting, jpeg), so gray, white and
    black need wide ranges or they end up labeled orange/red/brown
    '''
    hex = rgb_to_hex(img)
    hex_code = hex.lstrip('#')
    r, g, b = (int(hex_code[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360

    # achromatic cases first. Black clothes photograph as dark gray/brown, not pure black.
    # Very dark but strongly colored (navy, burgundy) keeps its color
    if (v < 0.22 and not (v >= 0.12 and s >= 0.6)) or (v < 0.35 and s < 0.25):
        return 'black'
    # Low saturation is a neutral. Dark shades show more tint so allow a bit more there
    if s < 0.18 or (s < 0.25 and v < 0.6):
        return 'white' if v > 0.75 else 'gray' # Shaded white fabric is still white

    # light yellows are muted too but still read as yellow, check them before brown
    if 40 <= h < 70 and v > 0.85 and s >= 0.25:
        return 'yellow'
    # brown: warm hues that are dark or muted. Beige, tan and camel go here,
    # only bright saturated warm colors are actually orange
    if 15 <= h < 50 and (v < 0.65 or s < 0.45):
        return 'brown'
    # muted yellows (khaki, sand) are brown too
    if 50 <= h < 70 and s < 0.35:
        return 'brown'

    red_hue = h < 15 or h >= 345
    # dark muted reds are brown (burgundy stays red since it is saturated)
    if red_hue and v < 0.45 and s < 0.5:
        return 'brown'
    # light, soft reds are pink
    if red_hue and s < 0.5 and v > 0.7:
        return 'pink'

    # dark magentas are purple or burgundy, not pink
    if 290 <= h < 345 and v < 0.55:
        return 'purple' if h < 325 else 'red'

    # hue bins
    bins = [
        (15, 'red'), (50, 'orange'), (70, 'yellow'),
        (165, 'green'), (200, 'cyan'), (260, 'blue'),
        (315, 'purple'), (345, 'pink'), (360, 'red'),
    ]
    for upper, name in bins:
        if h < upper:
            return name
    return 'red'
