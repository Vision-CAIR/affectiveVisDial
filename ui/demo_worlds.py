#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random 
import redis

from parlai.crowdsourcing.utils.worlds import CrowdOnboardWorld, CrowdTaskWorld  # type: ignore
from parlai.core.worlds import validate  # type: ignore
from joblib import Parallel, delayed  # type: ignore


class MultiAgentDialogOnboardWorld(CrowdOnboardWorld):
    def __init__(self, opt, agent):
        super().__init__(opt, agent)
        self.opt = opt

    def parley(self):
        self.agent.agent_id = "Onboarding Agent"
        self.agent.observe({"id": "System", "text": "Welcome onboard!"})
        x = self.agent.act(timeout=self.opt["turn_timeout"])
        self.agent.observe(
            {
                "id": "System",
                "text": "Thank you for your input! Please wait while "
                "we match you with another worker...",
                "episode_done": True,
            }
        )
        self.episodeDone = True


class MultiAgentDialogWorld(CrowdTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, image_data=None, shared=None):
        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("max_turns", 10)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.opt = opt

        img_url_default = 'https://dummyimage.com/600x400/000000/ffffff&text=ASK+Questions+about+the+image'
        self.emotion_labels = {
            'positive' : image_data['positive_emotion_label'],
            'negative' : image_data['negative_emotion_label']
        }

        self.image_data = image_data
        for idx, agent in enumerate(self.agents):
            agent.agent_id = f"Chat Agent {idx + 1}"
            agent.observe(
                {
                    "id": "System",
                    "text": 'You have been paired! Please, directly start with questions.<span style="color: red"> Do not start with words '
                    'like Hello, Hi, Hey, etc </span>' if not idx else 'You have been paired. Please, <span style="color: red"> avoid short answers e.g, YES/NO without further elaboration/details </span>',
                    "task_data": {
                        "image_src" : image_data['image_src'] if idx else img_url_default,
                        "positive_caption" : image_data['positive_caption'],
                        "positive_emotion_label": image_data['positive_emotion_label'],
                        "negative_caption" : image_data['negative_caption'],
                        "negative_emotion_label": image_data['negative_emotion_label'],
                        "positive_emoji_url" : image_data['positive_emoji_url'],
                        "negative_emoji_url" : image_data['negative_emoji_url']
                    }
                }
            )
    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        self.current_turns += 1
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=self.opt["turn_timeout"])
                if self.send_task_data:
                    acts[index].force_set(
                        "task_data",
                        {
                            "last_acting_agent": agent.agent_id,
                            "current_dialogue_turn": self.current_turns,
                            "utterance_count": self.current_turns + index,
                        },
                    )
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent
            if acts[index]["episode_done"]:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        if self.current_turns >= self.max_turns:
            self.episodeDone = True
            
            for idx, agent in enumerate(self.agents):
                if idx == 0:
                    agent.observe(
                        {
                            "id": "Coordinator",
                            "text": 'Please select one of the emotions below that reflects your imagination of the image that is shaped by your conversation with the fellow turker. Please, be specific and describe the justification in at least 10 words. \nPlease refer to pieces of information from the conversation you had with the Answerer that informed your decision.\n Please  avoid here responses that are irrelevant to our imagination of the artwork and how that constructed your emotion. \n For example, responses like "This person was a very good typist and conversationalist", "I enjoyed talking with a partner". etc',
                            "task_data": {
                                "positive_emotion_label": self.emotion_labels['positive'],
                                "negative_emotion_label": self.emotion_labels['negative'],
                                "respond_with_form": [
                                    {
                                        "type": "choices",
                                        "question": "Please, choose emotion based on the conversation",
                                        "choices": [
                                            "1",
                                        ],
                                    },
                                    
                                    {"type": "text", "question": 'Why/What makes you feel this particular emotion?'}
                                ]
                            },
                        }
                    )
                    agent.act()  # Request a response

            for idx, agent in enumerate(self.agents):  # Ensure you get the response
                if idx == 0:
                    form_result = agent.act(timeout=self.opt["turn_timeout"])
            
            for idx, agent in enumerate(self.agents):
                if idx==0:
                    agent.observe(
                        {
                            "id": "System",
                            "text": 'Congrats, now you can see the hidden image',
                            "task_data": {
                                "image_src" : self.image_data['image_src'],
                                "positive_caption" : self.image_data['positive_caption'],
                                "positive_emotion_label": self.image_data['positive_emotion_label'],
                                "negative_caption" : self.image_data['negative_caption'],
                                "negative_emotion_label": self.image_data['negative_emotion_label'],
                                "positive_emoji_url" : self.image_data['positive_emoji_url'],
                                "negative_emoji_url" : self.image_data['negative_emoji_url'],
                            }
                        }
                    )
                    agent.observe(
                            {
                                "id": "Coordinator",
                                "text": ' ',
                                "task_data": {
                                    "positive_emotion_label": self.emotion_labels['positive'],
                                    "negative_emotion_label": self.emotion_labels['negative'],
                                    "respond_with_form": [
                                        {
                                            "type": "choices",
                                            "question": "Please, choose emotion after observing the image",
                                            "choices": [
                                                "1",
                                            ],
                                        },
                                        {"type": "text", "question": 'If you selected different emotion label, What made you change your mind?'}
                                    ]
                                },
                            }
                        )
                    agent.act()  # Request a response

                if idx==1:
                    agent.observe(
                        {
                            "id": "System",
                            "text": 'Please, fill the form',
                            "task_data": {
                                "image_src" : self.image_data['image_src'],
                                "positive_caption" : self.image_data['positive_caption'],
                                "positive_emotion_label": self.image_data['positive_emotion_label'],
                                "negative_caption" : self.image_data['negative_caption'],
                                "negative_emotion_label": self.image_data['negative_emotion_label'],
                                "positive_emoji_url" : self.image_data['positive_emoji_url'],
                                "negative_emoji_url" : self.image_data['negative_emoji_url'],
                            }
                        }
                    )
                    agent.observe(
                            {
                                "id": "Coordinator",
                                "text": ' ',
                                "task_data": {
                                    "positive_emotion_label": self.emotion_labels['positive'],
                                    "negative_emotion_label": self.emotion_labels['negative'],
                                    "respond_with_form": [
                                        {
                                            "type": "choices",
                                            "question": "Please, choose emotion after observing the image",
                                            "choices": [
                                                "1",
                                            ],
                                        },
                                        {"type": "text", "question": 'Please, explain in 10 words what/why made you feel this way?'}
                                    ]
                                },
                            }
                        )
                    agent.act()  # Request a response

            for idx, agent in enumerate(self.agents):  # Ensure you get the response
                #if idx == 0:
                form_result = agent.act(timeout=self.opt["turn_timeout"])

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        return {"example_key": "example_value"}

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MultiAgentDialogOnboardWorld(opt, agent)


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    print(f"Validating onboarding data {data}")
    possible_answers = ["Agree", "agree", "AGREE"]
    answer = data['outputs']['messages'][2]['data']['text']
    is_correct = answer in possible_answers
    return is_correct


def make_world(opt, agents):
    # connect to Redis server
    r = redis.Redis(host='localhost', port=6379, password='', db=0, charset="utf-8", decode_responses=True)
    image_name = r.lpop('visdial_queue')
    #image_name = list(opt['image_data'].keys())[0]
    image_url = 'https://wikiart-dataset.s3.amazonaws.com/' + image_name + '.jpg'
    print("Num of images in REDIS: ",len(r.lrange('visdial_queue', 0, -1)))
    
    emojis = {
        'anger' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/anger-preview-rev-1-1.png",
        'disgust' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/disgust-trans.png",
        'fear' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/fear-removebg-preview.png",
        'sadness' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/sadness-removebg-preview.png",
        'excitement' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/excitement-transparent.png",
        'amusement' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/amusement-trans.png",
        'contentment': "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/contentment-trans.png",
        'awe' : "https://affective-dialog.s3.us-west-2.amazonaws.com/assets/emojis/awe-trans.png",
        'something else' : ''
    }
    caption_data = [
        (opt['image_data'][image_name]['positive'][-1], opt['image_data'][image_name]['positive'][0],
        emojis[opt['image_data'][image_name]['positive'][0]]),
        (opt['image_data'][image_name]['negative'][-1],opt['image_data'][image_name]['negative'][0],
        emojis[opt['image_data'][image_name]['negative'][0]])
    ]
    random.shuffle(caption_data)
    image_data = {
        "image_src" : image_url,
        "positive_caption" : caption_data[0][0],
        "positive_emotion_label": caption_data[0][1],
        "negative_caption" : caption_data[1][0],
        "negative_emotion_label": caption_data[1][1],
        "positive_emoji_url" : caption_data[0][2],
        "negative_emoji_url" : caption_data[1][2]
    }

    return MultiAgentDialogWorld(opt, agents, image_data)


def get_world_params():
    return {"agent_count": 2}
