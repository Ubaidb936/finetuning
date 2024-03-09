
##PromptFunc for Mistral 7B.........
def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    # Samples with additional context into.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    # Without context
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text


##PromptFunc for Zephr.........
def generate_prompt(datapoint):
    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:
    {context}
    
    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>
    {answer}
    </s>
     """
    prompt = prompt_template.format(
        context=datapoint["context"], 
        question=datapoint["question"], 
        answer=datapoint["answer"]
    )
    return prompt
    




def generate_prompt(data_point):
    
    if data_point['context']:
        prompt_template = """
        <|system|>
        Answer the question based on your knowledge. Use the following context to help:
        {context}
    
        </s>
        <|user|>
        {question}
        </s>
        <|assistant|>
        {answer}
        </s>
        """
        prompt = prompt_template.format(
        context=data_point["context"], 
        question=data_point["question"], 
        answer=data_point["answer"]
    )
        

    # Without context
    else:
        prompt_template = """
        <|system|>
        Answer the question based on your knowledge.
        </s>
        <|user|>
        {question}
        </s>
        <|assistant|>
        {answer}
        </s>
        """
        prompt = prompt_template.format(
        question=data_point["question"], 
        answer=data_point["answer"]
        )
    return prompt






