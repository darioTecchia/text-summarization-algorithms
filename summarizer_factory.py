from extractive.luhn.luhn import Luhn


def get_summarizer(algorithm):
    if(algorithm == 'bert'):
        from extractive.bert.Bert import BERT
        return BERT
    elif(algorithm == 'kl_sum'):
        from extractive.kl_sum.KlSum import KLSum
        return KLSum
    elif(algorithm == 'lexrank'):
        from extractive.lexrank.LexRank import LexRank
        return LexRank
    elif(algorithm == 'lsa'):
        from extractive.lsa.LSA import LSA
        return LSA
    elif(algorithm == 'luhn'):
        from extractive.luhn.Luhn import Luhn
        return Luhn
    elif(algorithm == 'bart'):
        from abstractive.BART.BART import BART
        return BART
    elif(algorithm == 't5'):
        from abstractive.T5.T5 import T5
        return T5
    else:
        raise Exception()