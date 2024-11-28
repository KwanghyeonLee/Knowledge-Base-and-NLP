import re
import csv
import jsonlines
from tqdm import tqdm

print("Preprocessing training files")
for date0 in ['0910','1011','1112']:
    with open(f"TWiki_Diffsets/wikipedia_{date0}_gpt2.csv", 'r') as f, jsonlines.open(f'./data/TemporalWiki/train/diffset_{date0}_filtered.jsonl', 'w') as writer:
        reader = csv.reader(f)

        reader.__next__()
        total=0
        real=0
        for line in tqdm(reader):
            total+=1
            # re.search
            # print(line)
            content = line[0].strip()
            if 'nan' in content and len(content)<50:
                continue

            if len(line[0]) < 50:
                continue
            
            letters = len(re.findall(r'[a-zA-Z]', line[0]))
            if letters/ len(line[0]) < 0.3:
                continue

            dic = {'text': line[0].strip()}
            writer.write(dic)
            real+=1

    print(total)
    print(real)

######
print("Preprocessing evaluation files")
for date0 in ['0901-1001', '1001-1101', '1101-1201']:
    with jsonlines.open(f"twiki_probes/{date0}_changed.jsonl", 'r') as ch, jsonlines.open(f"twiki_probes/{date0}_unchanged.jsonl", 'r') as unch:
        changed = list(ch)
        unchanged= list(unch)

    with jsonlines.open(f'data/TemporalWiki/eval/{date0}_changed.jsonl', 'w') as eval :
        for id in  range(len(changed)):
            eval.write(changed[id])

    with jsonlines.open(f'data/TemporalWiki/eval/{date0}_unchanged.jsonl', 'w') as eval :
        for id in  range(len(unchanged)):
            eval.write(unchanged[id])
