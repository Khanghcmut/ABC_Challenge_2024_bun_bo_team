from GCN.feeder import feeder_ABC

root_path = '/Users/vibuitruong/Documents/GitHub/ABC_Challenge_2024_bun_bo_team/data/Dataset-2'
dataset = feeder_ABC.Feeder(root_path)
for data, label, index in (dataset):
    print(data.shape)
    break