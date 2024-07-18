import json

with open("result.json", "r") as file:
    data = json.load(file)

filtered = []

print(f"Num of events: {len(data['traceEvents'])}")
for idx, i in enumerate(data["traceEvents"]):
    if (
        "torch._ops.aten." in i["name"]
        and i["pid"] == 692972
        # and i["ts"] > 1763677872547
        and i["ts"] > 1763678293465
        and i["ts"] < 1763679159195
    ):
        filtered.append(i)

print(filtered[0])

time = 0
for i in filtered:
    time += i["dur"]

print(time)
