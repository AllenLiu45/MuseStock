import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default='37',
                    help='Random seed')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max_epoch', type=int, default='300',
                    help='Training max epoch')
parser.add_argument('--patience', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='1',
                    help='GPU to use')
parser.add_argument('--batch_size', type=int, default='96',
                    help='Batch size')
parser.add_argument('--heads', type=int, default='4',
                    help='attention heads')
parser.add_argument('--hidn_att', type=int, default='60',
                    help='attention hidden nodes')
parser.add_argument('--hidn_rnn', type=int, default='360',
                    help='rnn hidden nodes')
parser.add_argument('--look_back_window', type=int, default='7',
                    help='look back window')
parser.add_argument('--dropout', type=float, default='0.2',
                    help='dropout rate')
parser.add_argument('--weight-constraint', type=float, default='0',
                    help='L2 weight constraint')
parser.add_argument('--clip', type=float, default='0.25',
                    help='rnn clip')
parser.add_argument('--infer', type=float, default='1',
                    help='if infer relation')
parser.add_argument('--relation', type=str, default='None',
                    help='all, competitor, customer, industry, stratigic, supply')
parser.add_argument('--save', type=bool, default=True,
                    help='save baseline_model')
parser.add_argument('--dataset', type=str, default='CMIN-US',
                    help='ACL18, CMIN-US, CMIN-CN')

args = parser.parse_args()