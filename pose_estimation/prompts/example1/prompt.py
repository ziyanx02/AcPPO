from .stage0.text import prompt as prompt0
from .stage1.text import prompt as prompt1
from .stage2.text import prompt as prompt2
from .stage3.text import prompt as prompt3
from .stage4.text import prompt as prompt4
from .stage5.text import prompt as prompt5
from .stage6.text import prompt as prompt6

prompt = prompt0 + prompt1 + prompt2 + prompt3 + prompt4 + prompt5 + prompt6