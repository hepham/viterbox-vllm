"""Vietnamese tokenizer for Viterbox TTS model"""
import logging
import os
from typing import List
from unicodedata import normalize

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer

# Special tokens (same as Viterbox original)
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


class ViterboxTokenizer(PreTrainedTokenizer):
    """
    A VLLM-compatible tokenizer for Viterbox Vietnamese TTS model.
    Based on the MTLTokenizer but simplified for Vietnamese.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file_path: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.check_vocabset_sot_eot()

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Instantiate a tokenizer from the Viterbox Vietnamese tokenizer file.
        """
        vocab_file = os.path.join(os.path.dirname(__file__), "tokenizer_vi_expanded.json")
        return cls(vocab_file_path=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"Missing {SOT} in vocab"
        assert EOT in voc, f"Missing {EOT} in vocab"

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def preprocess_text(self, raw_text: str, lowercase: bool = False, nfkd_normalize: bool = False):
        """
        Text preprocessor for Vietnamese.
        Viterbox does NOT lowercase or NFKD normalize Vietnamese text.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        # Parse out language token if it exists
        # Format: <vi>text or [vi]text
        language_id = None
        if text.startswith('<'):
            language_id = text.split('<')[1].split('>')[0]
            text = text.split('>')[1]
        
        # Preprocess text (no lowercase, no NFKD for Vietnamese)
        text = self.preprocess_text(text, lowercase=False, nfkd_normalize=False)
        
        # Prepend language token in [lang] format (Viterbox format)
        if language_id:
            text = f"[{language_id.lower()}] {text}"
        
        # Replace spaces with SPACE token
        text = text.replace(' ', SPACE)
        
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())
