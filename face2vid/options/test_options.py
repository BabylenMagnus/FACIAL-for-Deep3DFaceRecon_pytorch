### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--clip_length', type=int, default=500, help='length of generated clip')
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy',
                                 help='the path for clustered results of encoded features')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.parser.add_argument('--use_first_frame', action='store_true',
                                 help='If enabled, use first ground truth frame')
        self.parser.add_argument('--blink_path', type=str, default='../examples/test-result/obama2.npz')
        self.parser.add_argument('--test_id_name', type=str, default='obama2')
        self.isTrain = False
