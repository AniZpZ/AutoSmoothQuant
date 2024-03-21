import torch
from benchmarks.base import BaseLM
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
import torch
from autosmoothquant.utils import (
    get_model_architecture,
    get_config,
)


class quant_model(BaseLM):
    def __init__(
        self,
        pretrained="llama",
        tokenizer=None,
        quant_config=None,
        attn_implementation="eager",
        revision="main",
        subfolder=None,
        batch_size=1,
        dtype=torch.float16,
        device="cuda",
        max_length=-1,
    ):

        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.dtype = dtype
        config = get_config(model_path=pretrained)
        quant_model_class, _ = get_model_architecture(config)
        self.model = quant_model_class.from_pretrained(
            pretrained,
            quant_config=quant_config,
            device_map="auto",
            attn_implementation=attn_implementation,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            # torch_dtype=self.dtype
        )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     pretrained,
        #     revision=revision + ("/" + subfolder if subfolder is not None else ""),
        #     device_map='auto',
        #     torch_dtype=self.dtype
        # )
        if max_length != -1:
            self.model.config.max_sequence_length = max_length
        self.pretrained = pretrained
        self.no_split_modules = self.model._no_split_modules
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.max_sequence_length
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps, attention_mask=None):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps, attention_mask=attention_mask)[0][
                :, :, : len(self.tokenizer)
            ]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
