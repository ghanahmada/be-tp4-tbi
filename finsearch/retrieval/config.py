from dataclasses import dataclass


@dataclass
class ColBERTConfig:
    experiment_path: str = "/index/experiments/msmarco"
    index_name: str = "ir"
    index_url: str = "https://drive.google.com/uc?&id=1Y1Eee4mHQ-jDvBUNehFfEmMj-r9iNDcf"
    folder_name: str = "experiments"