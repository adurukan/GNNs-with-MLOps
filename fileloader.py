import json
import csv
from turtle import update
from data_schema import new_json
from dataclasses import asdict
from datetime import datetime


class FileLoader:
    def __init__(self, input_fn: str) -> None:
        self.input_fn = input_fn
        self.node_ids = []
        self.updated_jsons = []
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

    def recorder(self, node_id, data):
        node = None
        transaction_val = ""
        transaction_time = ""
        try:
            if str(node_id) == data["source"]:
                node = {data["target"]: "out"}
                transaction_val = data["value"]
                transaction_time = (
                    datetime.strptime(data["block_timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime(1970, 1, 1)
                ).total_seconds()
            elif str(node_id) == data["target"]:
                node = {data["source"]: "in"}
                transaction_val = data["value"]
                transaction_time = (
                    datetime.strptime(data["block_timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime(1970, 1, 1)
                ).total_seconds()

        except Exception:
            pass
        return node, transaction_val, transaction_time

    def _json(self):
        if self.load_from_csv() == True:
            csvs = []
            with open(self.input_fn, encoding="utf-8-sig") as csv_file:
                data_ = csv.DictReader(csv_file)
                for data in data_:
                    self.node_ids.append(data["source"])
                    self.node_ids.append(data["target"])
                    csvs.append(data)
                self.node_ids = list(set(self.node_ids))

                for node_id in self.node_ids:
                    nodes, transaction_vals, transaction_times, = (
                        [],
                        [],
                        [],
                    )
                    for data in csvs:
                        (
                            node,
                            transaction_val,
                            transaction_time,
                        ) = self.recorder(node_id, data)
                        if node is not None:
                            nodes.append(node)
                            transaction_times.append(transaction_time)
                            transaction_vals.append(transaction_val)

                    self.updated_jsons.append(
                        asdict(
                            new_json(
                                node_id=node_id,
                                node=nodes,
                                labels={"FA_1_case_1": False, "FA_1_case_2": False},
                                transaction_val=transaction_vals,
                                transaction_time=transaction_times,
                                further_attributes=[],
                            )
                        )
                    )

        elif self.load_from_json() == True:
            with open(self.input_fn, encoding="utf-8-sig") as json_file:
                data_ = json.load(json_file)
                for data in data_:
                    self.node_ids.append(data["source"])
                    self.node_ids.append(data["target"])
                self.node_ids = list(set(self.node_ids))

                for node_id in self.node_ids:
                    nodes, transaction_vals, transaction_times, = (
                        [],
                        [],
                        [],
                    )
                    for data in data_:
                        (
                            node,
                            transaction_val,
                            transaction_time,
                        ) = self.recorder(node_id, data)
                        if node is not None:
                            nodes.append(node)
                            transaction_times.append(transaction_time)
                            transaction_vals.append(transaction_val)

                    self.updated_jsons.append(
                        asdict(
                            new_json(
                                node_id=node_id,
                                node=nodes,
                                labels={"FA_1_case_1": False, "FA_1_case_2": False},
                                transaction_val=transaction_vals,
                                transaction_time=transaction_times,
                                further_attributes=[],
                            )
                        )
                    )

        else:
            pass

    def write(self, output_fn: str):
        with open(output_fn, "w") as outfile:
            for updated_json in self.updated_jsons:
                json.dump(updated_json, outfile, indent=2)
                outfile.write("\n")

    def convert_value(self, unit: str, factor: float) -> None:
        if unit == "EU":
            return factor * 1119.5
        elif unit == "USD":
            return factor * 1182.42
        pass

    def convert_nx_graph(self):
        # What is this function supposed to do?
        pass
