from __future__ import print_function, absolute_import

import argparse
import torch,time,os

torch.backends.cudnn.benchmark = True

from scripts.utils.misc import save_checkpoint, adjust_learning_rate

import scripts.datasets as datasets
import scripts.machines as machines
from options import Options

def main(args):
    
    if 'HFlickr' or 'HCOCO' or 'Hday2night' or 'HAdobe5k' in args.base_dir:
        dataset_func = datasets.BIH
    else:
        dataset_func = datasets.COCO

    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    lr = args.lr
    data_loaders = (train_loader,val_loader)

    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)
    print('============================ Initization Finish && Training Start =============================================')

    for epoch in range(Machine.args.start_epoch, Machine.args.epochs):

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        lr = adjust_learning_rate(data_loaders, Machine.optimizer, epoch, lr, args)

        Machine.record('lr',lr, epoch)        
        Machine.train(epoch)

        if args.freq < 0:
            Machine.validate(epoch)
            Machine.flush()
            Machine.save_checkpoint()

if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    args = parser.parse_args()
    print('==================================== WaterMark Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    print('==> {:50}: {:<}'.format("USE GPU",os.environ['CUDA_VISIBLE_DEVICES']))
    print('==================================== Stable Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) == parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Changed Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) != parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==================================== FINISH WITHOUT ERROR =============================================')
