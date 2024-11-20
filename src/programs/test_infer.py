import dspy
import dsp

from src.utils import extract_labels_from_strings
from .config import IreraConfig
from .signatures import InferSignatureEntity

import json


class RankSignatureEntity(dspy.Signature):
    __doc__ = f"""Given a snippet from a search result of entity, pick the 1 most applicable category from the options that is best expression of the entity. Options are given in the format 'depth1 > depth2, depth1 > depth2, ...', and the output must be one of these options without any modification."""

    text = dspy.InputField(prefix="Entity:")
    options = dspy.InputField(
        prefix="Options:",
        desc="List of comma-separated options for 1depth and 2depth categories to choose from. Format: 'depth1 > depth2, depth1 > depth2, ...'",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Category:",
        desc="depth1 > depth2 (without any modification, not bold, not italic, etc.)",
        format=lambda x: x,
    )
    

class Infer(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(InferSignatureEntity)

    def forward(self, text: str, options: list) -> dspy.Prediction:
        parsed_outputs = set()

        output = self.cot(text=text, options=options).completions.output
        print(output)

        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)

class Infer2(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(RankSignatureEntity)

    def forward(self, text: str, options: list) -> dspy.Prediction:
        parsed_outputs = set()

        output = self.cot(text=text, options=options).completions.output

        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)


if __name__ == "__main__":
    
    language = "ja"
    retriever_model_name = "jhgan/ko-sroberta-multitask" if language == "ko" else "colorfulscoop/sbert-base-ja"
    ontology_path = f"./data/entity/category_{language}.txt"
    
    with open(f"/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict_{language}.json", "r") as f:
        depth_dict = json.load(f)
        
    depth1_list = list(depth_dict.keys())
    
    with open(f"/data2/hy.jin/git/xmc.dspy/data/entity/category_{language}.txt", "w") as f:
        # depth1 > depth2
        for depth1 in depth1_list:
            depth2_list = depth_dict[depth1]
            for depth2 in depth2_list:
                f.write(f"{depth1} > {depth2}\n")
    
    import pandas as pd
    dataset_path = "./data/entity/all/ja_sample.csv"
    df = pd.read_csv(dataset_path)
    texts = df['text'].tolist()
    
    vllm_kwargs = {
        "temperature": 0.1,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "n": 1,
        "stop": []
    }
    
    # vllm = dspy.HFClientVLLM(model="google/gemma-2-9b-it", port=8002, url="http://10.10.210.100", **vllm_kwargs)
    # dspy.settings.configure(lm=vllm)
    
    
    config = IreraConfig.from_dict({
        "infer_signature_name": "infer_entity",
        "rank_signature_name": "rank_entity",
        "prior_A": 0,
        "prior_path": "./data/entity/entity_priors.json",
        "rank_topk": 5,
        "chunk_context_window": 3000,
        "chunk_max_windows": 5,
        "chunk_window_overlap": 0.02,
        "rank_skip": False,
        "ontology_path": ontology_path,
        "ontology_name": "entity",
        "retriever_model_name": retriever_model_name,
        "optimizer_name": "left-to-right"
    })
    infer = Infer(config)
    infer2 = Infer2(config)
    
    from src.programs import InferRetrieveRank
    from dspy import Example
    program = InferRetrieveRank.load(path=f"./results/entity_category_infer-retrieve-rank_02_{language}/program_state.json")
    
    for text in texts[1:2]:
        example = Example(text=text).with_inputs('text')
        prediction = program(**example.inputs())
        # print(vllm.inspect_history(n=1))
        print(prediction)
        
        # print(text)
        # pred = infer(text, depth1_list).predictions.pop()
        # print(pred)
        # print(vllm.inspect_history(n=1))
        # # depth1 = pred.split(" > ")[0]
        # # options = [f"{depth1} > {depth2}" for depth2 in depth_dict[depth1]]
        # # depth2 = infer2(text, options).predictions.pop()
        # # print(depth2)
        # # print(vllm.inspect_history(n=1))
        # # print("===")