def get_vocab(all_data, start_ch="."):
    """
    Given a list of names returns all the unique characters in a set
    Includes "." as the start character
    """
    vocab = set("".join(all_data))
    vocab.add(start_ch)
    return vocab


def get_encode_decode(vocab):
    """
    Given the vocab returns 2 functions to encode and decode each chaacter
    """
    stoi = {ch:i for i, ch in enumerate(sorted(vocab))}
    itos = {i:ch for ch,i in stoi.items()}

    encode = lambda name: [stoi[ch] for ch in name] if len(name) > 1 else stoi[name]
    decode = lambda ints: "".join([itos[i] for i in ints]) if isinstance(ints, list) is True else itos[ints]

    return encode, decode



def get_samples(data, encode, context_size=3, start_ch="."):
    """
    given a list of names and an encode function, create training samples using the given context_size
    The outputs are Python lists, not tensors
    """
    xs, ys = [], []
    for name in data:
        context = [start_ch] * context_size
        for ch in name:
            xs.append(encode(context))
            ys.append(encode(ch))

            context.pop(0)
            context.append(ch)
    return xs, ys
