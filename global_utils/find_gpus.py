import os 
def find_gpus(nums=2):
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/.tmp_free_gpus")

    with open(os.path.expanduser ('~/.tmp_free_gpus') , 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                                for idx,x in enumerate(frees) ]
    idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freeMemory_pair[:nums] ]
    usingGPUs =  ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs

