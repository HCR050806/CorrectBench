from .rci import RCI
from .selfrefine import SELF_REFINE
from .cove import CoVe

def create_method(args, model, task=None):
    """
    Factory method to create a method instance for self-correction
    """
    if args.method == 'rci':
        return RCI(model, task, args.prompting_style)
    elif args.method == 'selfrefine':
        return SELF_REFINE(model, task, init_prompt_examples_file = '/Self-Correction-Benchmark/method/init.txt', fb_prompt_examples_file = '/home/yueke/correct/hcr_selfcorrection/Self-Correction-Benchmark/method/feedback.txt')
    elif args.method == 'cove':
        return CoVe(model, task)
    else:
        raise ValueError('Unknown method: {}'.format(args.method))