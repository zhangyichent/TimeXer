from data_provider.data_loader import Dataset_Custom,Dataset_ETT_hour
from torch.utils.data import DataLoader
import os
import pickle

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'weather': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    cache_path = args.cache_path
    cache_file = os.path.join(cache_path, flag + '.pkl')
    if os.path.exists(cache_file) and args.load_pickle:
        f = open(cache_file, "rb")
        data_set = pickle.load(f)
        f.close()
    else:
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq
        )
        with open(cache_file, "wb") as f:
            pickle.dump(data_set, f)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
