import json
import csv
from data_schema import new_json
from dataclasses import asdict


class FileLoader:
    def __init__(self, input_fn: str) -> None:
        self.input_fn = input_fn
        pass

    def load_from_csv(self) -> bool:
        if self.input_fn[-3:] == "csv":
            return True
        else:
            return False

    def load_from_json(self) -> bool:
        if self.input_fn[-4:] == "json":
            return True
        else:
            return False

    def _json(self):
        if self.load_from_csv() == True:
            with open(self.input_fn, encoding="utf-8") as csv_file:
                data_ = csv.DictReader(csv_file)
                print(f"data_: {data_}")
        elif self.load_from_json() == True:
            with open(self.input_fn, encoding="utf-8") as json_file:
                data_ = json.load(json_file)
                print(f"data_: {data_}")
                # Modification of json file
                updated_json = new_json(**data_)
                print(f"updated_json: {asdict(updated_json)}")
        else:
            pass

    def write(self, output_fn: str):
        pass

    def convert_value(self, unit: str, factor: float) -> None:
        pass

    def convert_nx_graph(self):
        pass


###Testing
path = "test.json"
fileloader = FileLoader(path)
fileloader._json()
