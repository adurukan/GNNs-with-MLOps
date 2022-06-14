from data_schema import new_json
import json


def read_json(path):
    f = open(path, "r")
    data = json.load(f)
    data = json.dumps(data, indent=2)
    f.close()
    return data


path = "test.json"
print(path[-4:])
data = read_json("test.json")
print(data)
