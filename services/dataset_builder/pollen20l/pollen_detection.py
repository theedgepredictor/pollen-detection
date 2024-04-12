import collections
import json
import os

import datasets


_HOMEPAGE = "https://www.kaggle.com/datasets/nataliakhanzhina/pollen20ldet"
_LICENSE = "CC BY 4.0"
_CITATION = """\
@misc{ pollen20ldet,
    title = { Combating data incompetence in pollen images detection and classification for pollinosis prevention },
    type = { Open Source Dataset },
    author = { Khanzhina, Natalia and Filchenkov, Andrey and Minaeva, Natalia and Novoselova, Larisa and Petukhov, Maxim and Kharisova, Irina and Pinaeva, Julia and Zamorin, Georgiy and Putin, Evgeny and Zamyatina, Elena and others},
    howpublished = { \\url{ https://www.kaggle.com/datasets/nataliakhanzhina/pollen20ldet } },
    url = { https://www.kaggle.com/datasets/nataliakhanzhina/pollen20ldet },
    journal = { Computers in biology and medicine },
    volume={140},
    pages={105064},
    publisher = { Elsevier },
    year = { 2022 },
}
"""

### I want to look at multiple ways to load categories. Pollen has multiple classification levels and is usually classified via latin version of the term

_CATEGORIES = [
    'buckwheat',
    'clover',
    'angelica',
    'angelica_garden',
    'willow',
    'hill_mustard',
    'linden',
    'meadow_pink',
    'alder',
    'birch',
    'fireweed',
    'nettle',
    'pigweed',
    'plantain',
    'sorrel',
    'grass',
    'pine',
    'maple',
    'hazel',
    'mugwort'
]

_ANNOTATION_FILENAME = "_annotations.json"


class POLLENDETECTIONConfig(datasets.BuilderConfig):
    """Builder Config for pollen-detection"""

    def __init__(self, data_urls, **kwargs):
        """
        BuilderConfig for pollen-detection.
        Args:
          data_urls: `dict`, name to url to download the zip file from.
          **kwargs: keyword arguments forwarded to super.
        """
        super(POLLENDETECTIONConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_urls = data_urls


class POLLENDETECTION(datasets.GeneratorBasedBuilder):
    """pollen-detection object detection dataset"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        POLLENDETECTIONConfig(
            name="full",
            description="Full version of pollen-detection dataset.",
            data_urls={
                "train": "https://huggingface.co/datasets/Charliesgt/Pollen20LDet/resolve/main/data/train.zip",
                "valid": "https://huggingface.co/datasets/Charliesgt/Pollen20LDet/resolve/main/data/valid.zip",
                "test": "https://huggingface.co/datasets/Charliesgt/Pollen20LDet/resolve/main/data/test.zip",
            },
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(self.config.data_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "folder_dir": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "folder_dir": data_files["valid"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "folder_dir": data_files["test"],
                },
            ),
]

    def _generate_examples(self, folder_dir):
        def process_annot(annot, category_id_to_category):
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "category": category_id_to_category[annot["category_id"]],
            }

        image_id_to_image = {}
        idx = 0

        annotation_filepath = os.path.join(folder_dir, _ANNOTATION_FILENAME)
        with open(annotation_filepath, "r") as f:
            annotations = json.load(f)
        category_id_to_category = {category["id"]: category["name"] for category in annotations["categories"]}
        image_id_to_annotations = collections.defaultdict(list)
        for annot in annotations["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        filename_to_image = {image["file_name"]: image for image in annotations["images"]}

        for filename in os.listdir(folder_dir):
            filepath = os.path.join(folder_dir, filename)
            if filename in filename_to_image:
                image = filename_to_image[filename]
                objects = [
                    process_annot(annot, category_id_to_category) for annot in image_id_to_annotations[image["id"]]
                ]
                with open(filepath, "rb") as f:
                    image_bytes = f.read()
                yield idx, {
                    "image_id": image["id"],
                    "image": {"path": filepath, "bytes": image_bytes},
                    "width": image["width"],
                    "height": image["height"],
                    "objects": objects,
                }
                idx += 1

