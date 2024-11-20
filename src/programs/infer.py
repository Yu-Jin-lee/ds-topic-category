import dspy
import json

from src.utils import extract_labels_from_strings
from .config import IreraConfig
from .signatures import supported_signatures


class Infer(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(
            supported_signatures[config.infer_signature_name]
        )
        self.depth1_list = self._load_depth1_list()
        
    def _load_depth1_list(self):
        if self.config.language == "ja":
            with open("/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict_ja.json", "r") as f:
                depth_dict = json.load(f)
        else:
            with open("/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict.json", "r") as f:
                depth_dict = json.load(f)
        
        return list(depth_dict.keys())

    def forward(self, text: str) -> dspy.Prediction:
        parsed_outputs = set()

        output = self.cot(text=text, options=self.depth1_list).completions.output
        
        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)
