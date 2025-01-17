import json
import boto3
import json_repair
from termcolor import colored
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from lib import frames
from lib import frame_utils

def make_transcript(vtt_file):
    with open(vtt_file, encoding="utf-8") as f:
        transcript = f.read()
    
    return {
        'role': 'user',
        'content': [
            {"text":'Here is the transcripts in <transcript> tag:\n<transcript>{0}\n</transcript>\n'.format(transcript)}
        ]
    }


def make_conversation_message(text):
    message = {
        'role': 'user',
        'content': [
            {'text':'No conversation.'}
        ]
    }

    if text:
        message['content'] = 'Here is the conversation of the scene in <conversation> tag.\n<conversation>\n{0}\n</conversation>\n'.format(text)

    return message


def make_conversation_example():
    example = {
        'topics': [
            {
                'start': '00:00:10.000',
                'end': '00:00:32.000',
                'reason': 'It appears the topic talks about...'
            }
        ]
    }

    return {
        'role': 'user',
        'content': [
            {'text':'JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))}
        ]
    }

def analyze_conversations(model_id, transcript_file):

    response = {}
    messages = []

    # transcript
    transcript_message = make_transcript(transcript_file)
    messages.append(transcript_message)

    # output format?
    messages.append({
        'role': 'assistant',
        'content': [{'text':'Got the transcript. What output format?'}]
    })

    # example output
    example_message = make_conversation_example()
    messages.append(example_message)

    # prefill output
    messages.append({
        'role': 'assistant',
        'content': [
            {'text':'{'}
        ]
    })

    ## system prompt to role play
    system = [{'text':'''
    You are a media operation assistant who analyses movie transcripts in WebVTT format
    and suggest topic points based on the topic changes in the conversations. 
    It is important to read the entire transcript.
    '''}]
    
    infParams = {"maxTokens": 4096, "topP": 0.7, "temperature": 0.1}
    
    try:
        response['response'] = inference(system, 
                                         messages, 
                                         model_id, 
                                         infParams)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response['response'] = inference(system, 
                                 messages, 
                                 model_id, 
                                 infParams)
    return response

def inference(system, messages, model_id, infParams):

    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    model_response = bedrock_runtime_client.converse(
        modelId=model_id, 
        messages=messages, 
        system=system, 
        inferenceConfig=infParams
    )

    response_body = model_response["output"]["message"]

    # patch the json string output with '{' and parse it
    response_content = response_body['content'][0]['text']
    if response_content[0] != '{':
        response_content = '{' + response_content

    try:
        response_content = json.loads(response_content)
    except Exception as e:
        print(colored("Malformed JSON response. Try to repair it...", 'red'))
        try:
            response_content = json_repair.loads(response_content, strict=False)
        except Exception as e:
            print(colored("Failed to repair the JSON response...", 'red'))
            print(colored(response_content, 'red'))
            raise e

    response_body['content'][0]['json'] = response_content
    response_body['usage'] = model_response["usage"]

    return response_body

def get_contextual_information(model_id, images, text, iab_definitions):
    system = [{'text':'''You are a media operation engineer. Your job is to review a clip from a video 
    content presented in a sequence of consecutive images. Each image
    contains a sequence of frames presented in a 4x7 grid reading from left to
    right and then from top to bottom. Interpret the frames as the time 
    progression of a video clip.  Don't refer to specific frames, instead, think
    about what is happening over time in the scene.  You may also optionally be given the
    conversation of the scene you can use to understand the context of
    the scene. 

    You are asked to provide the following information: a detailed 
    description to describe the scene using the visual and audio, identify the most relevant IAB taxonomy, 
    GARM, sentiment, and brands and logos that 
    may appear in the scene, and five most relevant tags from the scene.
    
    It is important to return the results in JSON format and also includes a
    confidence score from 0 to 100. Skip any explanation.
    '''}]

    messages = []
 
    # adding sequences of composite images to the prompt.  Limit is 20.
    message_images = make_image_message(images[:19])
    messages.append(message_images)

    # adding the conversation to the prompt
    messages.append({
        'role': 'assistant',
        'content': [{'text':'Got the images. Do you have the conversation of the scene?'}]
    })
    message_conversation = make_conversation_message(text)
    messages.append(message_conversation)

    # other information
    messages.append({
        'role': 'assistant',
        'content': [{'text':'OK. Do you have other information to provdie?'}]
    })

    other_information = []
    ## iab taxonomy
    iab_list = make_iab_taxonomoies(iab_definitions['tier1'])
    other_information.append(iab_list)

    ## GARM
    garm_list = make_garm_taxonomoies()
    other_information.append(garm_list)

    ## Sentiment
    sentiment_list = make_sentiments()
    other_information.append(sentiment_list)

    messages.append({
        'role': 'user',
        'content': other_information
    })

    # output format
    messages.append({
        'role': 'assistant',
        'content': [{'text':'OK. What output format?'}]
    })
    output_format = make_output_example()
    messages.append(output_format)

    # prefill '{'
    messages.append({
        'role': 'assistant',
        'content': [{'text':'{'}]
    })
    
    infParams = {"maxTokens": 4096, "topP": 0.7, "temperature": 0.1}

    try:
        response = inference(system, 
                             messages, 
                             model_id, 
                             infParams)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(system, 
                             messages, 
                             model_id, 
                             infParams)

    return response

def display_prompt(model_params):
    print (f'MODEL_ID: {MODEL_ID}\n')
    print (f'System Prompt:\n\n{model_params["system"]}')
    print (f'Messages:\n\n')
    for message in model_params['messages']:
        print (json.dumps(message))

    print('\n')

    return
    


def display_conversation_cost(response, pricing=(0,0)):
    # us-east-1 pricing
    input_per_1k, output_per_1k = pricing

    if 'input_tokens' in response['usage']:
        input_tokens = response['usage']['input_tokens']
    else:
        input_tokens = response['usage']['inputTokens']

    if 'input_tokens' in response['usage']:
        output_tokens = response['usage']['output_tokens']
    else:
        output_tokens = response['usage']['outputTokens']
        
    conversation_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${conversation_cost}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': conversation_cost,
    }

def display_contextual_cost(usage, pricing=(0,0)):
    # us-east-1 pricing
    input_per_1k, output_per_1k = pricing

    if 'input_tokens' in usage:
        input_tokens = usage['input_tokens']
    else:
        input_tokens = usage['inputTokens']

    if 'input_tokens' in usage:
        output_tokens = usage['output_tokens']
    else:
        output_tokens = usage['outputTokens']

    contextual_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${round(contextual_cost, 4)}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': contextual_cost,
    }


def make_image_message(composite_images):
    # adding the composite image sequences
    image_contents = [{'text': 'Here are {0} images containing frame sequence that describes a scene.'.format(len(composite_images))
    }]

    for image in composite_images:
        with open(image['file'], "rb") as image_file:
            image_data = image_file.read()
            image_contents.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": image_data,
                    },
                }
            })

    return {
        'role': 'user',
        'content': image_contents
    }

def make_conversation_message(text):
    message = {
        'role': 'user',
        'content': [{'text':'No conversation.'}]
    }

    if text:
        message['content'][0]['text'] = f'''
            Here is the conversation of the scene.
            
            **conversation:**
            { text }
            
            '''

    return message

def make_iab_taxonomoies(iab_list):

    iab=""
    for item in iab_list:
        iab += f"- {item['name']}\n"
        
    iab += "- None\n"

    
    iab = [item['name'] for item in iab_list]
    iab.append('None')

    return {'text': f'''
                    Here is a list of IAB Taxonomies. Only answer 
                    the IAB taxonomy from this list.
                    
                    **IAB taxonomy:**
                    { iab }

                    '''}

def make_garm_taxonomoies():
    garm = f'''
            Here is a list of GARM Taxonomies in <garm> tag. Only answer
            the GARM taxonomy from this list.
            
            **GARM taxonomy:**
            - Adult & Explicit Sexual Content
            - Arms & Ammunition
            - Crime & Harmful acts to individuals and Society, Human Right Violations
            - Death, Injury or Military Conflict
            - Online piracy
            - Hate speech & acts of aggression
            - Obscenity and Profanity, including language, gestures, and explicitly gory, graphic or repulsive content intended to shock and disgust
            - Illegal Drugs, Tobacco, ecigarettes, Vaping, or Alcohol
            - Spam or Harmful Content
            - Terrorism
            - Debated Sensitive Social Issue
            - None
            
            '''

    return {'text': garm}

def make_sentiments():
    sentiments = f'''
        Here is a list of Sentiments in <sentiment> tag. Only answer the
        sentiment from this list:

        **sentiment:**
        - Positive
        - Neutral
        - Negative
        - None

        '''

    return {'text': sentiments}

def make_output_example():
    example = {
        'description': {
            'text': 'The scene describes...',
            'score': 98
        },
        'sentiment': {
            'text': 'Positive',
            'score': 90
        },
        'iab_taxonomy': {
            'text': 'Station Wagon',
            'score': 80
        },
        'garm_taxonomy': {
            'text': 'Online piracy',
            'score': 90
        },
        'brands_and_logos': [
            {
                'text': 'Amazon',
                'score': 95
            },
            {
                'text': 'Nike',
                'score': 85
            }
        ],
        'relevant_tags': [
            {
                'text': 'BMW',
                'score': 95
            }
        ]            
    }
    
    return {
        'role': 'user',
        'content': [{'text':'Return JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))}]
    }

