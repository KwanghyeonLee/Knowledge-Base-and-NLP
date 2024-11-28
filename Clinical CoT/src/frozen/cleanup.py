import json
import argparse

'''
This script is used to split prediction.json file into test.json and test_AIBL.json file
'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='model/prediction/json/data/path')
    parser.add_argument('--org_path', type=str, default='ADNI/json/data/path')
    parser.add_argument('--org_AIBL_path', type=str, default='AIBL/json/data/path')
    parser.add_argument('--output_path', type=str, default='output/ADNI/json/test/path')
    args = parser.parse_args()
    return args

def main(args):
    # open input_path
    data = []
    with open(args.input_path, 'r') as file:
        for line in file:
            temp = json.loads(line)
            for i in range(len(temp['predictions'])):
                input_temp = {
                    "input": temp['inputs'][i],
                    "prediction": temp['predictions'][i]
                }
                data.append(input_temp)
    
    # open org_path
    with open(args.org_path, 'r') as file:
        org_data = json.load(file)
        if 'data' in org_data:
            org_data = org_data['data']
    
    save_data = []

    for d in org_data:
        dict_to_save = {
            'id': d['id'],
            'input': d['input'],
            'label': d['label'],
            'patient_data': d['patient_data'],
            'org_label': d['org_label'],
        }
        for j in range(len(data)):
            if data[j]['input'] == d['input']:
                dict_to_save['prediction'] = data[j]['prediction']
                break
        save_data.append(dict_to_save)

    assert len(save_data) == len(org_data)
    
    # save output_path
    with open(args.output_path, 'w') as file:
        json.dump(save_data, file, indent=4)
        
    # open org_AIBL_path
    with open(args.org_AIBL_path, 'r') as file:
        org_data2 = json.load(file)
        if 'data' in org_data2:
            org_data2 = org_data2['data']
    
    save_data2 = []
    for d in org_data2:
        dict_to_save = {
            'id': d['id'],
            'input': d['input'],
            'label': d['label'],
            'patient_data': d['patient_data'],
            'org_label': d['org_label'],
        }
        for j in range(len(data)):
            if data[j]['input'] == d['input']:
                dict_to_save['prediction'] = data[j]['prediction']
                break
        save_data2.append(dict_to_save)
        
    assert len(save_data2) == len(org_data2)
    
    with open(args.output_path.replace('test.json', 'test_AIBL.json'), 'w') as file:
        json.dump(save_data2, file, indent=4)

if __name__ == '__main__':
    args = get_args()
    main(args)