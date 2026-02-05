import os
import yaml
import pandas as pd

target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mean-pooling")

rows = []


for name in os.listdir(target_dir):

    path = os.path.join(target_dir, name)
    if not os.path.isdir(path):
            continue
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.yaml'):
                yaml_path = os.path.join(path, file)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                model_layer = data["model"]["model_layer"]
                print(model_layer)
    with open(os.path.join(path, f'dev.uuas'), 'r') as f:
        uuas = f.read().strip()
        print(uuas)
    with open(os.path.join(path, f'dev.spearmanr-5_50-mean'), 'r') as f:
        spearmanr = f.read().strip()
        print(spearmanr)
    rows.append({
        'model_layer': model_layer,
        'uuas': uuas,
        'spearmanr': spearmanr,
    })
df = pd.DataFrame(rows)
df = df.sort_values(by='model_layer')
df.to_csv(os.path.join(target_dir, 'summary_results.csv'), index=False)
        