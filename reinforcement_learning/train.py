import torch
import random
from model import MMFAgent
from argparse import ArgumentParser
from pytorch_lightning import Trainer


def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    model = None
    if args.device == 'cuda':
        assert torch.cuda.is_available()
    model = MMFAgent(input_dim=args.input_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers, L=args.L, k=args.k, matrix_size=args.matrix_size,
                    n_batch_per_epoch=args.n_batch_per_epoch, train_batch_size=args.batch_size, device=args.device, lr=args.lr).to(args.device)
    trainer = Trainer(max_epochs=args.max_epochs,
                      gradient_clip_val=args.gradient_clip_val)
    trainer.fit(model)
    trainer.save_checkpoint(f'mmf_ep{trainer.current_epoch}.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()

    # random seed
    parser.add_argument('--random_seed', type=int, default=1234)

    # hparams
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=int, default=1)

    # network structure
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--matrix_size', type=int, default=30)

    # train set
    parser.add_argument('--n_batch_per_epoch', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=470)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)

    # device
    parser.add_argument('--device', type=str, default='cpu')  # 'cpu' or 'cuda'

    args = parser.parse_args()
    main(args)
