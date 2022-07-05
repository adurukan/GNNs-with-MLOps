import ujson as json
import pandas as pd

class FA1_detection:
    def __init__(self, input_fn: str, unit: str) -> None:
        self.input_fn = input_fn
        self.unit = unit
        pass

    def read_ndjson(self) -> pd.DataFrame:
        records = map(json.loads, open(self.input_fn))
        return pd.DataFrame.from_records(records)
    
    def convert_value(self) -> None:
        if self.unit == "EU":
            return 1119.5
        elif self.unit == "USD":
            return 1182.42
        pass

    def preprocessing(self) -> pd.DataFrame:
        df = self.read_ndjson()
        neighbor_nodes = []
        for col1, row in zip(df["node_id"], df["node"]):
            for d in row:
                for k, v in d.items():
                    neighbor_nodes.append({"node_id": col1, "neighbor_nodes": k, "direction": v})
        df_1 = pd.DataFrame(neighbor_nodes)

        transaction_time = []
        transaction_list = []
        for col1, col2 in zip(df["transaction_val"], df["transaction_time"]):
            for c1 in col1:
                transaction_list.append(c1)
            for c2 in col2:
                transaction_time.append(c2)
        df_1["transaction_val"] = transaction_list
        df_1["transaction_time"] = transaction_time

        df["FA_1_case_1"] = df["labels"].apply(lambda x: x["FA_1_case_1"])
        df["FA_1_case_2"] = df["labels"].apply(lambda x: x["FA_1_case_2"]) 

        df_final = pd.merge(df_1, df[["node_id","FA_1_case_1", "FA_1_case_2"]], on="node_id", how="left")

        df_final["transaction_val"] = df_final["transaction_val"] * self.convert_value()

        return df_final

    def create_mask(self, df):
        # filter out every transation that is over 10000
        df = df[(df["transaction_val"] < 10000) & (df["transaction_val"] != 0.00)]

        # filter out single transation
        tmp = (df.groupby(["node_id","direction"]).count() == 1).neighbor_nodes.reset_index().rename(columns={"neighbor_nodes":"single_trans"})
        tmp = tmp.loc[tmp.single_trans == False].drop(["single_trans"], axis=1)

        # create the df_mask dataframe
        # mask_1_case_1 checks if it's a "out" relation 
        # mask_1_case_2 checks if it's a "in" relation

        df_mask = pd.merge(tmp, df, on = ["node_id","direction"], how="left")


        df_mask["mask_1_case_1"] = df_mask["direction"].apply(lambda x: True if x == "out" else False)
        df_mask["mask_1_case_2"] = df_mask["direction"].apply(lambda x: True if x == "in" else False)

        # filter out all nodes, which has total transaction amount < 10000 

        tmp_2 = df_mask.groupby(["node_id","direction"]).sum().transaction_val.reset_index()
        tmp_2["threshold_tot_trans_val"] = tmp_2["transaction_val"] > 10000
        df_mask = pd.merge(df_mask, tmp_2.drop(["transaction_val"],axis=1), on = ["node_id","direction"], how="left")

        df_mask["mask_2_case_1"] = df_mask["mask_1_case_1"] & df_mask["threshold_tot_trans_val"]
        df_mask["mask_2_case_2"] = df_mask["mask_1_case_2"] & df_mask["threshold_tot_trans_val"]

        return df_mask
    
    def detection_FA1(self, df_node, start_index) -> bool:
        node_flag = False
        end_index = df_node.shape[0]
        time_window_start = df_node['transaction_time'].values[start_index]
        time_window_end =  time_window_start + 172801
        index_latest_trx = df_node['transaction_time'].searchsorted(time_window_end, side = 'left') - 1
        trx_sum = df_node['transaction_val'].iloc[start_index:index_latest_trx + 1].sum()
        # amount threshold
        if trx_sum >= 10000:
            node_flag = True
            return node_flag
        # early-stopping
        elif (index_latest_trx == end_index):
            return node_flag
        else:
            # check for duplicated timestamp to avoid unnecessary looping 
            duplicated_ts = df_node[df_node['transaction_time'] == time_window_start]
            start_index = start_index + len(duplicated_ts)
            if start_index == end_index:
                node_flag = False
            else:
                node_flag = self.detection_FA1(df_node, start_index)
        return node_flag

    def flagger_FA1(self, df_candidates):
        start_index = 0
        node_flags = {}
        #loop over all candidates
        for node in df_candidates['node_id'].unique().tolist():
            df_node = df_candidates[df_candidates['node_id'] == node].sort_values(by='transaction_time').reset_index(drop=True)
            node_flag = self.detection_FA1(df_node, start_index)
            node_flags[node] = node_flag
        return node_flags

if __name__ == "__main__":
    path_json = "data/new.json"
    fa1_detection = FA1_detection(path_json, "USD")
    df_final = fa1_detection.preprocessing()
    df_mask = fa1_detection.create_mask(df_final)

    case1_candidates = df_mask[df_mask['mask_2_case_1'] == True]
    case2_candidates = df_mask[df_mask['mask_2_case_2'] == True]
    
    case1_flags = fa1_detection.flagger_FA1(case1_candidates)
    case2_flags = fa1_detection.flagger_FA1(case2_candidates)
    print(case1_flags)
    print(case2_flags)

    dummy_flags = [False] * df_final.shape[0]
    df_final['Case1_Flag'] = dummy_flags
    df_final['Case2_Flag'] = dummy_flags
    for node in df_final['node_id']:
        #mask = (int(df_final['node_id']) == node)
        #print(node)
        if node in case1_flags:
            df_final.loc[(df_final['node_id'] == node, 'Case1_Flag')] = case1_flags[node]
        if node in case2_flags:
            df_final.loc[(df_final['node_id'] == node, 'Case2_Flag')] = case2_flags[node]  
    
