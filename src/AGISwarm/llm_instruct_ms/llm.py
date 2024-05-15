""" LLM Instruct Model Inference """

from threading import Thread
from typing import cast
from time import perf_counter
import torch
import transformers
from vllm import LLM, SamplingParams


SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-8b-Instruct",
    "DevsDoCode/LLama-3-8b-Uncensored",
    "DevsDoCode/LLama-3-8b-Uncensored-4bit",
    "IlyaGusev/saiga_llama3_8b",
    "RichardErkhov/IlyaGusev_-_saiga_llama3_8b-4bits",
]


MODEL_IS_4bit = {
    "meta-llama/Meta-Llama-3-8B-Instruct": False,
    "unsloth/llama-3-8b-Instruct-bnb-4bit": True,
    "unsloth/llama-3-8b-Instruct": False,
    "DevsDoCode/LLama-3-8b-Uncensored": False,
    "DevsDoCode/LLama-3-8b-Uncensored-4bit": True,
    "IlyaGusev/saiga_llama3_8b": False,
    "RichardErkhov/IlyaGusev_-_saiga_llama3_8b-4bits": True,
}

BNB_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class MyLLM:
    """LLM Instruct Model Inference"""

    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_llama3_8b",
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):

        self.pipeline = cast(
            transformers.TextGenerationPipeline,
            transformers.pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                tokenizer=tokenizer_name,
                model_kwargs={
                    "quantization_config": (
                        BNB_CONFIG if MODEL_IS_4bit[model_name] else None
                    )
                },
            ),
        )
        self.streamer = transformers.TextIteratorStreamer(
            self.pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
        )

    @staticmethod
    def prepare_prompt(
        tokenizer, messages: list[dict], reply_prefix: str = None
    ):
        """Prepare prompt for model"""
        prompt = cast(
            str,
            tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ),
        ) + ("$" if reply_prefix else "")
        if reply_prefix:
            prompt = prompt.replace(
                "<|eot_id|>$", "<|eot_id|>assistant\n\n" + reply_prefix
            )

        return prompt

    def postprocess(self, llm_output: dict):
        """Postprocess text"""
        answer = llm_output[0]["generated_text"].replace("\\n", "\n")
        return answer

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        print_output: bool = True,
    ):
        """Generate text from prompt"""
        generated_text = ""
        for new_text in self.generate_yields(
            messages,
            max_new_tokens=max_new_tokens,
            reply_prefix=reply_prefix,
            temperature=temperature,
            top_p=top_p,
        ):
            generated_text += new_text
            if print_output:
                print(new_text, end="")
        return generated_text

    def generate_yields(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        penalty_alpha: float = 0.1,
    ):
        """Generate text from prompt"""
        prompt = self.prepare_prompt(self.pipeline.tokenizer, messages, reply_prefix)
        thread = Thread(
            target=self.pipeline,
            kwargs=dict(
                text_inputs=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                streamer=self.streamer,
                clean_up_tokenization_spaces=True,
                penalty_alpha=penalty_alpha,
            ),
        )
        thread.start()
        for new_text in self.streamer:
            yield new_text


class HFLLM:
    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_llama3_8b",
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BNB_CONFIG if MODEL_IS_4bit[model_name] else None,
            device_map="auto",
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        penalty_alpha: float = 0.1,
    ):
        """Generate text from prompt"""
        prompt = MyLLM.prepare_prompt(self.tokenizer, messages, reply_prefix)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=len(inputs["input_ids"]) + max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            penalty_alpha=penalty_alpha,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class VLLM:
    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_llama3_8b",
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LLM(model=model_name, dtype=torch.float16, tensor_parallel_size=2)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        penalty_alpha: float = 0.1,
    ):
        """Generate text from prompt"""
        prompt = MyLLM.prepare_prompt(self.tokenizer, messages, reply_prefix)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=penalty_alpha,
        )
        return self.model.generate(prompt, sampling_params)


if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello!"}]
    # tm = perf_counter()
    # llm = HFLLM("unsloth/llama-3-8b-Instruct")
    # print(llm.generate(messages))
    # print(f"Time: {perf_counter() - tm:.2f}s")
    tm = perf_counter()
    llm = VLLM("unsloth/llama-3-8b-Instruct")
    print(llm.generate(messages))
    print(f"Time: {perf_counter() - tm:.2f}s")
