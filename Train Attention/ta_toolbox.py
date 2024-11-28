import torch

def make_conv(sessions0, names0, ses_id, lim0=float('inf')):
    text0 = "This is a dialogue "
    for traj in sessions0[ses_id]['dialogue']:
        name = names0[traj['speaker']]
        utter = traj['text']
        text0 += f"### {name} : {utter} "
        if len(text0) > lim0:
            break
    text0 = text0.replace('  ',' ').strip()
    return text0



def giveme_label_mask(query_token, label_token):  # query_token : {input_ids: [[123123]]},  label_token : [123123]
    
    query_token = query_token.input_ids[0]
    pallet = torch.zeros(len(query_token))
    # label_token = label_token.input_ids[0][1:]
    for start0 in list(range(len(pallet)- len(label_token)+1))[::-1]:
        if torch.equal(query_token[start0: start0+len(label_token)], label_token):
            pallet[start0: start0+len(label_token)] =1 
            break
    return pallet