import argparse
import json
#import tqdm
import os

with open("combine.json.gz", "r") as fp:
    combine_file = json.load(fp)

json_files_path = os.getcwd()+"/jsons/"

jsons = os.listdir(json_files_path)

json_strings = []

for files in jsons:
    with open(json_files_path+"/"+files, "r") as fp:
        json_strings.append(json.load(fp))


counter = 1
for vidx in range(len(combine_file["results"]["videos"])):
    for cidx in range(len(combine_file["results"]["videos"][vidx]["clips"])):
        for preidx in range(len (combine_file["results"]["videos"][vidx]["clips"][cidx]["predictions"])  ):
            # print(combine_file["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"].keys())
            for qidx in combine_file["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"].keys()   :
                for json_string in json_strings:
                    # print(json_string["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"])
                    # print(combine_file["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"])
                    try:
                        if len(json_string["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"][str(qidx)]["bboxes"])>0:
                            
                            combine_file["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"][qidx] = json_string["results"]["videos"][vidx]["clips"][cidx]["predictions"][preidx]["query_sets"][qidx]


                        
                            break
                    except:
                        pass

with open("combine.json", "w") as outfile:
    json.dump(combine_file, outfile)


