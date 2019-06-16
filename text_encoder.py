import random
from collections import Counter
from reserved_tokens import SOS_INDEX
from reserved_tokens import EOS_INDEX
from reserved_tokens import UNKNOWN_INDEX
from reserved_tokens import RESERVED_ITOS
from reserved_tokens import PADDING_INDEX


class JapaneseTextEncoder:
    """ Encodes the text using a tokenizer.

    Args:
        corpus (list of strings): Text strings to build dictionary on.
        min_occurrences (int, optional): Minimum number of occurences for a token to be
            added to dictionary.
        append_sos (bool, optional): If 'True' append SOS token onto the begin to the encoded vector.
        append_eos (bool, optional): If 'True' append EOS token onto the end to the encoded vector.
        padding (bool, optional): If 'True' pad a sequence.
        filters (list of strings): Part of Speech strings to remove.
        neologd ({0, 1}, optional): 0 for original MeCab; otherwise NEologd.
        reserved_tokens (list of str, optional): Tokens added to dictionary; reserving the first
            'len(reserved_tokens') indices.

    Example:
        >>> corpus = ["セネガルつええ、ボルト三体くらいいるわ笑笑", \
                      "しょーみコロンビアより強い", \
                      "それなまちがいないわ"]

        >>> encoder = JapaneseTextEncoder(
                corpus,
                append_eos=True,
                padding=True
            )
        >>> encoder.encode("コロンビア強い")
        [18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', 'セネガル', 'つえ', 'え', '、', 'ボルト', '三', '体', 'くらい', ' いる', 'わ', '笑', 'しょ', 'ー', 'み', 'コロンビア', 'より', '強い', 'それ', 'な', 'まちがい', 'ない']
        >>> encoder.decode(encoder.encode("コロンビア強い"))
        コロンビア強い</s>

        >>> encoder.dataset
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 2],
         [15, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 2],
         [21, 22, 23, 24, 13, 0, 0, 0, 0, 0, 0, 0, 2]]

    """

    def __init__(self,
                 corpus,
                 min_occurrences=1,
                 append_sos=False,
                 append_eos=False,
                 padding=False,
                 filters=None,
                 neologd=0,
                 maxlen=None,
                 reserved_tokens=RESERVED_ITOS):
        try:
            import MeCab
        except ImportError:
            print("Please install MeCab.")
            raise

        if not isinstance(corpus, list):
            raise TypeError("Corpus must be a list of strings.")

        if neologd:
            self.tagger = MeCab.Tagger(r"-Ochasen -d C:\neologd")
        else:
            self.tagger = MeCab.Tagger("-Ochasen")
        self.corpus = corpus  # sentence of list
        self.append_sos = append_sos
        self.append_eos = append_eos
        self.padding = padding
        self.tokens = Counter()
        self.filters = ["BOS/EOS"]
        if filters is not None:
            if not isinstance(filters, list):
                raise TypeError("Filters must be a list of POS.")
            self.filters += filters

        self.maxlen = 0  # length of a sequence
        for sentence in self.corpus:
            tokens = self.tokenize(sentence)
            if tokens:
                self.tokens.update(tokens)
                self.maxlen = max(self.maxlen, len(tokens))
        if maxlen:
            self.maxlen = maxlen

        self.itos = reserved_tokens.copy()
        self.stoi = {token: index for index, token in enumerate(reserved_tokens)}
        for token, cnt in self.tokens.items():
            if cnt >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

        self.dataset = None  # e.x. [[0, 1, 2], [3, 4, 2]]

    @property
    def vocab(self):
        return self.itos

    @property
    def word2id(self):
        return self.stoi

    @property
    def id2word(self):
        return {index: token for token, index in self.stoi.items()}

    def build(self):
        self.dataset = [self.encode(sentence) for sentence in self.corpus]

    def build_seq2seq(self, corpus):
        """ Corpus must be list of lists.
            Each contained lists have two sentences message and reply.
            [
                ['message_1', 'reply_1'],
                ['message_2', 'reply_2']
            ]
        """
        self.dataset = [[self.encode(message), self.encode(reply)] for message, reply in corpus]

    def encode(self, sentence, sos_index=SOS_INDEX, eos_index=EOS_INDEX, unknown_index=UNKNOWN_INDEX,
               padding_index=PADDING_INDEX):
        tokens = self.tokenize(sentence)
        if tokens is None:
            raise TypeError("Invalid type None...")
        indices = [self.stoi.get(token, unknown_index) for token in tokens]
        if self.append_sos:
            indices.insert(0, sos_index)
        if self.append_eos:
            indices.append(eos_index)
        if self.padding:
            indices += [padding_index] * (self.maxlen - len(indices))
        return indices

    def decode(self, indices):
        tokens = [self.itos[index] for index in indices]
        tokens = list(filter(lambda x: x != "<pad>", tokens))
        return "".join(tokens)

    def get_batch_dataset(self, size=50, shuffle=True):
        batch_dataset = []
        if shuffle:
            self.shuffle()
        for i in range(0, len(self.dataset), size):
            start = i
            end = start + size
            batch_dataset.append(self.dataset[start:end])
        return batch_dataset

    def shuffle(self):
        random.shuffle(self.dataset)

    def tokenize(self, sentence):
        self.tagger.parse("")
        tag = self.tagger.parseToNode(sentence)
        tokens = []
        while tag:
            features = tag.feature.split(",")
            pos = features[0]
            token = tag.surface
            if pos in self.filters:
                tag = tag.next
                continue
            tokens.append(token)
            tag = tag.next
        return tokens if tokens else None
