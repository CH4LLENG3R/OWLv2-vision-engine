import PIL
import requests
from PIL import Image
import torch

from typing import Union, List

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from common.singleton import SingletonMeta
from logging import getLogger

logger = getLogger(__name__)


class OWLv2(metaclass=SingletonMeta):
    def __init__(self, device="cuda", threshold=0.2):
        self._threshold = threshold
        self._device = device
        self._processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self._model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=device)

    @staticmethod
    def _format_results(results):
        scores = results[0]["scores"].cpu().detach().numpy().astype(float).tolist()
        boxes = results[0]["boxes"].cpu().detach().numpy().astype(int).tolist()
        formatted_results = []
        for i, text_label in enumerate(results[0]["text_labels"]):
            formatted_results.append({"box": boxes[i], "score": scores[i], "label": text_label})
        return formatted_results

    def predict(self, image: PIL.Image.Image, text_labels: Union[str | List[str]]):
        text_labels = [text_labels]
        inputs = self._processor(text=text_labels, images=image, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([(image.height, image.width)])

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self._threshold, text_labels=text_labels
        )

        return OWLv2._format_results(results)
