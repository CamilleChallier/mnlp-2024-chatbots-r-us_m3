from transformers import AutoTokenizer
import json


model_checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def read_jsonl_file(file_path):
    """
    Function to read a jsonl file.

    Args:
        file_path: path to the jsonl file

    Returns:
        read_data: list of dictionaries
    """
    read_data = []
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            read_data.append(json_object)
    return read_data


def epfl_course_split_dataset(dataset):
    """
    Function to split the epfl course dataset into questions, answers, and courses.

    Args:
        dataset: list of dictionaries with keys "question", "answer", and "course"

    Returns:
        questions: list of questions
        answers: list of answers
        courses: list of courses 
    """
    questions = []
    answers = []
    courses = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answer"])
        courses.append(data["course"])

    return questions, answers, courses


def split_test_dataset(dataset):
    """
    Function to split the test dataset into questions and answers.

    Args:
        dataset: list of dictionaries with keys "question" and "answers"

    Returns:
        questions: list of questions
        answers: list of answers (list of lists)
    """
    questions = []
    answers = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answers"])

    return questions, answers


def mcqa_split_dataset(dataset):
    """
    Function to split the MCQA datasets into questions, answers, and subjects.

    Args:
        dataset: list of dictionaries with keys "question", "answer" (and "subject")

    Returns:
        questions: list of questions
        answers: list of answers
        subjects: list of subjects 
    """
    # check if the dataset has 'subject' as key
    if "subject" in dataset[0].keys():
        subjects = []
        questions = []
        answers = []
        for data in dataset:
            questions.append(data["question"])
            answers.append(data["answer"])
            subjects.append(data["subject"])

        return questions, answers, subjects
    # if not, return only questions and answers
    else:
        questions = []
        answers = []
        for data in dataset:
            questions.append(data["question"])
            answers.append(data["answer"])

        return questions, answers
    

def create_prompting_inputs(questions, courses, answers, tokenizer):
    """
    Function to create the prompting inputs for the model.

    Args:
        questions: list containing the questions of the dataset
        courses: list containing the courses of the dataset
        answers: list containing the answers of the dataset
        tokenizer: tokenizer 

    Returns:
        inputs: list containing the prompting inputs
        labels: list containing the labels
    """
    inputs = []
    labels = []
    for i in range(len(questions)):
        # promting taken from the template of flan T5
        #? https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
        prompt = "Question: " + questions[i] + "Answer: "
        inputs.append(prompt)
        labels.append(answers[i])

    return inputs, labels


def create_prompting_inputs_test_dataset(questions, answers, tokenizer):
    """
    Function to create the prompting inputs for the model for the test dataset.

    Args:
        questions: list containing the questions of the test dataset
        answers: list containing the answers of the test dataset
        tokenizer: tokenizer 

    Returns:
        inputs: list containing the prompting inputs
        labels: list containing the labels
    """
    inputs = []
    labels = []
    for i in range(len(questions)):
        # promting taken from the template of flan T5
        #? https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
        prompt = "Question:\n" + questions[i] + "\nAnswer: "
        inputs.append(prompt)
        labels.append(answers[i])

    return inputs, labels


def mcqa_create_prompting_inputs(questions, answers):
    """
    Function to create the prompting inputs for the model for the MCQA datasets.

    Args:
        questions: list containing the questions of the dataset 
        answers: list containing the answers of the dataset, only a letter per answer

    Returns:
        inputs: list containing the prompting inputs
        labels: list containing the labels
    """
    inputs = []
    labels = []
    for i in range(len(questions)):
        # promting taken from the template of flan T5
        #? https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
        # ("{text}\n\nWhat's the best answer to this question: {question}?\n\n{options_}", "{answer}")
        prompt = "Considering the different options, what's the best answer to this question: \n" + questions[i] + "?\n\nLetter of the answer: " 
        inputs.append(prompt)
        labels.append(answers[i])

    return inputs, labels


def mask_padtokens_labels(labels, tokenizer):
    """
    Function to mask the padding tokens in the labels.

    Args:
        labels: tensor of labels
        tokenizer: tokenizer 

    Returns:
        labels_masked: tensor of labels with padding tokens masked
    """
    labels_masked = labels
    # maks token -> -100
    labels_masked[labels == tokenizer.pad_token_id] = -100

    return labels_masked


def stem_split_dataset(dataset, test_size=0.2, random_state=7):
    """
    Function to split the stem dataset into questions, answers, and courses.

    Args:
        dataset: list of dictionaries with keys "question", "answer", and "subject"

    Returns:
        questions: list of questions
        answers: list of answers
        courses: list of courses 
    """
    questions = []
    answers = []
    courses = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answer"])
        courses.append(data["subject"])

    return questions, answers, courses


