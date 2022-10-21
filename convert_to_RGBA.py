from logging import exception
import PIL
from PIL import Image
import os
from pathlib import Path

original_path = "./original_dataset"
rgba_path = "original_dataset_rgba"

Path(rgba_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(rgba_path, "black")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(rgba_path, "blue")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(rgba_path, "green")).mkdir(parents=True, exist_ok=True)

for path, _, images in os.walk(original_path):

    print("Num images in folder {}: {}".format(path, len(images)))

    for image in images:
        full_path = os.path.join(path, image)
        try:

            im = Image.open(full_path)
            _class = full_path.split("/")[2] + "/"
            image_no_ext = image.split(".")[0]

            im.convert("RGBA").save(f"./" + rgba_path + "/" +
                                    _class+image_no_ext+"_rgba.png")

        except Exception as e:
            print(e)
