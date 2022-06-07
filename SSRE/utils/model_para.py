def filter_para(model, args, lr):
    if args.opt == 'opt1':
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]



