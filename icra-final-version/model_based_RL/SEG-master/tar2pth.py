import torch
import collections

infile_path = 'model_best.pth.tar'
outfile_path = 'sedla34.pth'
tar = torch.load(infile_path)
gpu = tar['state_dict']
cpu = collections.OrderedDict()
for key in gpu:
    cpu['.'.join(key.split('.')[1:])] = gpu[key].cpu()
torch.save(cpu, outfile_path)


# infile_path = 'output/inat17_dla102x2_512se/inat17_dla102x2_best.pth.tar'
# outfile_path = 'inat17_sedla102x2.pth'
# # outfile_path = infile_path.split('/')[-1].replace('_best.pth.tar', '.pth')
# tar = torch.load(infile_path)
# gpu = tar['state_dict']
# cpu = collections.OrderedDict()
# for key in gpu:
#     cpu['.'.join(key.split('.')[1:])] = gpu[key].cpu()
# torch.save(cpu, outfile_path)

# infile_path = 'output/inat18_dla102x2_512se/inat18_dla102x2_best.pth.tar'
# outfile_path = 'inat18_sedla102x2.pth'
# tar = torch.load(infile_path)
# gpu = tar['state_dict']
# cpu = collections.OrderedDict()
# for key in gpu:
#     cpu['.'.join(key.split('.')[1:])] = gpu[key].cpu()
# torch.save(cpu, outfile_path)
