from nflows import transforms
from torch.nn import functional as F


def spline_flow(inp_dim, nodes, num_blocks=2, nstack=3, tail_bound=None, tails=None, activation=F.relu, lu=0,
                num_bins=10):
    transform_list = []
    for i in range(nstack):
        # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
        # the final layer
        # TODO: Number of bins increase?
        tpass = tails
        if tails:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes, num_blocks=num_blocks,
                                                                               tail_bound=tb, num_bins=num_bins,
                                                                               tails=tpass, activation=activation)]
        if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
            transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])
