import json
import random
import math

def subsample_dataset(in_path, out_path, sample_pct=0.001,seed_val=1):
    random.seed(1) # Set random seed for reproducibility
    with open(in_path, "r") as f:
        obj = json.load(f)
    obj["videos"] = [
        {
            **v,
            "clips": [
                {
                    **c,
                    "annotations": random.sample(
                        c["annotations"], math.ceil(len(c["annotations"]) * sample_pct)
                    ),
                }
                for c in random.sample(
                    v["clips"], math.ceil(len(v["clips"]) * sample_pct),
                )
            ],
        }
        for v in random.sample(
            obj["videos"], math.ceil(len(obj["videos"]) * sample_pct)
        )
    ]

    with open(out_path, "w") as outfile:
      json.dump(obj, outfile)

datasets = [
    ("vq_val_1.json", "/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/vq2d_cvpr/data/vq_val.json"),
    #("vq_train_1.json", "/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/vq2d_cvpr/data/vq_train.json"),
    # ("/content/ego4d_data/v1/annotations/vq_test_unannotated.json", "/content/episodic-memory/VQ2D/data/vq_test_unannotated.json")
]
for pair in datasets:
  subsample_dataset(pair[0], pair[1])

# Write sampled video_uids into a file so we can download them as a subset
video_uids = []
clip_uids = []
for file in [pair[1] for pair in datasets]:
  with open(file) as f:
    obj = json.load(f)
    video_uids.extend([v['video_uid'] for v in obj['videos']])
    clip_uids.extend([c['clip_uid'] for v in obj['videos'] for c in v['clips']])

video_uids = list(set(video_uids))
with open('sampled_video_uids.txt', 'w') as f:
  for uid in video_uids:
    f.write(uid + "\n")

with open('sampled_clip_uids.txt', 'w') as f:
  for uid in clip_uids:
    f.write(uid + "\n")

print("Sampled Videos:", video_uids)
print("Sampled Clips:", clip_uids)
