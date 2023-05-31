import json

file = json.load(open("vq_test_unannotated.json"))

ids = []
for i in range(len(file["videos"])):
    ids.append(file["videos"][i]["video_uid"])

with open("test_uids.txt", "w") as f:
    for i in ids:
        f.write( i + '\n')