# import json
# import os

# with open("/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/valid_e3.json") as f:
#     valid = json.load(f)

# with open("/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF/checkpoint/2022-10-11-11-20-43-HFM-5-att-128-1-bert-base-vit_b_32-/dev_bad_case.txt") as f:
#     bad = f.readlines()

# with open("/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/emoji_mapping.json") as f:
#     emap = json.load(f)

# emojis = {}
# for l in valid:
#     if "emoji_" in l["text"]:
#         emojis[l["id"]] = l["text"]

# bad_em = {}
# for l in bad:
#     id_ , gt, pred = l.strip().split(",")
#     if id_ in emojis:
#         bad_em[id_] = (emojis[id_], gt)

# image_root = "/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/dataset_image/"
# output_root = "/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/embad/"
# print(len(bad_em.keys()))
# for id_, ti in bad_em.items():
#     text,gt = ti
#     text = text.split(" ")
#     image_p = image_root + id_ + ".jpg"
#     os.system(f"cp {image_p} {output_root}")
#     print("gt " , gt)
#     print(" ".join(text))
#     for token in text:
#         if "emoji_" in token and token in emap:
#             print(emap[token])
#     print("=========================")
#     while True:
#         aaa = input()
#         if aaa != "p":
#             break
#         else:
#             exit()
import json
# id2text = {}
# with open("/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/clean_image_text.txt") as f:
#     for l in f.readlines():
#         l = l.strip().split(" ")
#         id_ = l[0]
#         text = " ".join(l[1:])
#         id2text[id_] = text

# with open("/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/clean_image_text.json","w") as f:
#     f.write(json.dumps(id2text))
def read_json(p):
    with open(p) as f:
        content = json.load(f)
    res = {}
    for l in content:
        res[l["id"]] = l
    return res


root = "/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF1/dataset/data/HFM/"
for sp in ["train","valid","test"]:
    print(sp)
    output_p = f"{sp}_e4.json"
    input1_p = f"{sp}_e3.json"
    input2_p = f"{sp}_m.json"
    input3_p = f"{sp}_e.json"
    input1 = read_json(root + input1_p) # with emoji
    input2 = read_json(root + input2_p)
    input3 = read_json(root + input3_p)
    output = []
    for id_, l in input1.items():
        i2 = input2[id_]
        i3 = input3[id_]
        l["text"] = i2["text"]
        l["emoji"] = i3["emoji"]
        output.append(l)
    with open(root+output_p,"w") as f:
        f.write(json.dumps(output))
        
