from fileloader import FileLoader

path_json = "data/2021-1000.json"
path_csv = "data/2021-1000.csv"

file_loader = FileLoader(input_fn=path_csv)
print(file_loader.load_from_csv())

file_loader._json()
file_loader.write("data/new.json")

print(f"0.056 ether in USD: {file_loader.convert_value('USD', 0.056)}")
