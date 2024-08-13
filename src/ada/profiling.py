import json


def _separate_by_thread(data):
    # find set of unique pids under data["traceEvents"][idx]["pid"]
    pids = set()
    for idx in range(len(data["traceEvents"])):
        pids.add(data["traceEvents"][idx]["pid"])

    data_per_pid = {}
    for pid in pids:
        data_per_pid[pid] = {
            "traceEvents": [],
            "viztracer_metadata": data["viztracer_metadata"],
            "file_info": data["file_info"],
        }

    for idx in range(len(data["traceEvents"])):
        pid = data["traceEvents"][idx]["pid"]
        data_per_pid[pid]["traceEvents"].append(data["traceEvents"][idx])

    # now dump to separate json files
    for pid in pids:
        with open(f"result_{pid}.json", "w") as file:
            json.dump(data_per_pid[pid], file)


if __name__ == "__main__":
    with open("result.json", "r") as file:
        data = json.load(file)

    _separate_by_thread(data)
