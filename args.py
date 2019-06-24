def init_parser(parser):
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--recording', action='store_true')
    parser.add_argument('--video-folder', type=str, default='videos')
    parser.add_argument('--use-guidance', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-supervision', action='store_true')
    parser.add_argument('--frame-height', type=int, default=256)
    parser.add_argument('--frame-width', type=int, default=256)

    parser.add_argument('--time-decay', type=float, default=0.97)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--expert-bar', type=int, default=50)
    parser.add_argument('--expert-ratio', type=float, default=0.05)
    parser.add_argument('--safe-length-collision', type=int, default=5)
    parser.add_argument('--safe-length-offroad', type=int, default=5)
    parser.add_argument('--bin-divide', type=list, default=[5, 5])

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--frame-history-len', type=int, default=3)
    parser.add_argument('--pred-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--save-path', type=str, default='spc')
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--num-total-act', type=int, default=2)
    parser.add_argument('--epsilon-frames', type=int, default=50000)
    parser.add_argument('--learning-freq', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num-train-steps', type=int, default=10)
    # enviroument configurations
    parser.add_argument('--env', type=str, default='torcs')
    parser.add_argument('--server', type=bool, default=False)

    # model configurations
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--drn-model', type=str, default='dla46x_c')
    parser.add_argument('--classes', type=int, default=4)

    parser.add_argument('--use-collision', action='store_true')
    parser.add_argument('--use-offroad', action='store_true')
    parser.add_argument('--use-speed', action='store_true')

    parser.add_argument('--sample-with-offroad', action='store_true')
    parser.add_argument('--sample-with-collision', action='store_true')
    parser.add_argument('--speed-threshold', type=float, default=20)
    parser.add_argument('--port', type=int, default=2000)


def post_processing(args):
    args.env = args.env.lower()
    args.save_path = '{0}_{1}_{2}'.format(args.save_path, args.env, args.pred_step)
    args.sync = 'torcs' in args.env or 'carla' in args.env
    return args
