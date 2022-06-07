from utils.model.module.classifier import fc, cosine, euclidean, ncc, nce, nce_adt, fc_IL, fc_IL_base
from torchvision.models import resnet


def build_classifier(args, outplanes):
    if args.classifier == 'fc':
        return fc.LinearClassifier(outplanes, args)
    elif args.classifier == 'fc_IL':
        return fc_IL.LinearClassifier(outplanes, args)
    elif args.classifier == 'fc_IL_base':
        return fc_IL_base.LinearClassifier(outplanes, args)
    elif args.classifier == 'cosine':
        return cosine.CosineClassifier(outplanes, args)
    elif args.classifier == 'euc':
        return euclidean.EUCClassifier(outplanes, args)
    elif args.classifier == 'ncc':
        return ncc.NCCClassifier(outplanes, args)
    elif args.classifier == 'nce':
        return nce.NCEClassifier(outplanes, args)
    elif args.classifier == 'nce_adt':
        return nce_adt.NCEClassifier_adt(outplanes, args)
    else:
        raise NotImplementedError