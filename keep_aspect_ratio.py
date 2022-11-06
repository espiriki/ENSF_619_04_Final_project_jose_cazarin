import torch
from PIL import Image


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class PadToMaintainAR(torch.nn.Module):

    def __init__(self, aspect_ratio):
        super().__init__()
        self.aspect_ratio = aspect_ratio

    def forward(self, img):

        current_aspect_ratio = img.size[0] / img.size[1]
        target_aspect_ratio = self.aspect_ratio
        original_width = img.size[0]
        original_height = img.size[1]
        new_img = []

        if current_aspect_ratio == target_aspect_ratio:
            new_img = img
        if current_aspect_ratio < target_aspect_ratio:
            # need to increase width
            target_width = int(target_aspect_ratio * original_height)
            pad_amount_pixels = target_width - original_width
            new_img = add_margin(img, 0, int(pad_amount_pixels/2),
                                 0, int(pad_amount_pixels/2), (0, 0, 0))

        if current_aspect_ratio > target_aspect_ratio:
            # need to increase height
            target_height = int(original_width/target_aspect_ratio)
            pad_amount_pixels = target_height - original_height
            new_img = add_margin(img, int(pad_amount_pixels/2),
                                 0, int(pad_amount_pixels/2), 0, (0, 0, 0))

        return new_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{str(self.aspect_ratio)}"
