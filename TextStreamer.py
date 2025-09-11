from transformers import TextStreamer


class TextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = True, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt  # 是否打印prompt
        self.decode_kwargs = decode_kwargs  # 解码参数
        # 用于记录流式输出过程中的变量
        self.token_cache = []   # 缓存token
        self.print_len = 0       # 记录上次打印位置
        self.next_tokens_are_prompt = True  # 第一次为True，后续为False，记录当前调用put()时是否为prompt

    def put(self, value):
        """
        传入token后解码，然后在他们形成一个完整的词时将其打印到标准输出stdout
        """
        # print(self.tokenizer.decode(value.tolist(), skip_special_tokens=True))
        # print(value)
        # 这个类只支持 batch_size=1
        # 第一次运行.put()时，value=input_id，此时检测batch大小，input_id.shape：(batch_size, seq_len)
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        # 如果输入batch形式，但是batch_size=1，取第一个batch序列
        elif len(value.shape) > 1:
            value = value[0]

        # 第一次输入的视为prompt，用参数判断是否打印prompt
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # 将新token添加到缓存，并解码整个token
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # 如果token以换行符结尾，则清空缓存
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # 如果最后一个token是中日韩越统一表意文字，则打印该字符
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # 否则，打印直到最后一个空格字符（简单启发式，防止输出token是不完整的单词，在前一个词解码完毕后在打印）
        # text="Hello!"，此时不打印。text="Hello! I"，打印Hello!
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        # self.on_finalized_text(printable_text)
        print(printable_text,flush=True,end="")

    def end(self):
        """清空缓存，并打印换行符到标准输出stdout"""
        # 如果缓存不为空，则解码缓存，并打印直到最后一个空格字符
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # flush=True，立即刷新缓冲区，实时显示，取消缓冲存在的延迟
        # 如果stream_end为True，则打印换行符
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """检查CP是否是CJK字符"""
        # 这个定义了一个"chinese character"为CJK Unicode块中的任何内容：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)

        # 我们使用Unicode块定义，因为这些字符是唯一的，并且它们是所有主要语言的常见字符。
        # 注意，CJK Unicode块不仅仅是日语和韩语字符，
        # 尽管它的名字如此，现代韩语的Hangul字母是另一个块，
        # 日语的Hiragana和Katakana也是另一个块，
        # 那些字母用于写space-separated words，所以它们不被特别处理，像其他语言一样处理
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):
            return True

        return False
