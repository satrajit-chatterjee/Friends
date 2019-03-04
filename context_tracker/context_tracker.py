import json
import torch
import torch.nn
import numpy as np
import context_tracker.salience as salience
import sentiment_analysis.model as sen_model
import sentiment_analysis.train as utils

GLOVE_PATH = ""
context_hist = []


class ContextTracker:
    def __init__(self):
        self.context_hist = []
        self.sentence = ""  # this needs to be replaced by sentence input from the mobile app

    def context_history(self):
        net_input = ContextTracker.sentence_processor()

        net_utilities = utils.create_emb_layer()
        emotion_net = sen_model.Net(net_utilities, num_classes=7)
        act_net = sen_model.Net(net_utilities, num_classes=4)

        emotion = emotion_net(net_input)
        act = act_net(net_input)

        sal_finder = salience.SalienceFinder(e_tag=emotion, a_tag=act)
        google_response_string = ""
        max_key, max_val = sal_finder.process(google_response_string)
        word_2_vec = ContextTracker.load_glove_model(GLOVE_PATH)

        if not self.context_hist:
            self.context_hist.append(max_key)
        else:
            if ContextTracker.cosine_distance(self.context_hist[-1], max_key, word_2_vec) > 0.5:
                flag = False
                for x in enumerate(self.context_hist[:-1]):
                    if ContextTracker.cosine_distance(self, x, max_key, word_2_vec) < 0.5:
                        self.context_hist.append(x)
                        flag = True
                if not flag:
                    self.context_hist.append(max_key)
            else:
                self.context_hist.append(self.context_hist[-1])
        if len(self.context_hist) > 50:
            self.context_hist = self.context_hist[25:-1]
        return self.context_hist, emotion, act, word_2_vec

    def sentence_processor(self):
        word2vec = ContextTracker.load_glove_model(GLOVE_PATH)
        weighted_sentence = []
        sentence = self.sentence.split()
        for j, i in enumerate(sentence):
            i = i.lower()
            if i == "?" or i == "," or i == "." or i == "!" or i == ";" or i == "'":
                sentence.remove(i)
        if len(sentence) > 278:
            sentence = sentence[:278]
        else:
            while len(sentence) != 278:
                sentence.append("<PAD>")
        for j, i in enumerate(sentence):
            try:
                weighted_sentence[j] = word2vec[i]
            except KeyError:
                if i == "<PAD>":
                    weighted_sentence[j] = np.zeros((50,))
                else:
                    weighted_sentence[j] = np.random.normal(scale=0.6, size=(50,))
        weighted_sentence = np.array(weighted_sentence)
        weighted_sentence = torch.from_numpy(weighted_sentence)
        return weighted_sentence

    def tensor_builder(self):
        weighted_sentence = ContextTracker.sentence_processor()
        con_hist, emo, act, word2vec = ContextTracker.context_history(self)
        w = np.sqrt(sum(weighted_sentence ** 2))
        weighted_sentence = weighted_sentence / w
        weighted_sentence = torch.from_numpy(weighted_sentence)
        act = list([[act]]) * 50
        emo = list([[emo]]) * 50
        weighted_sentence = torch.cat([weighted_sentence, emo, act], dim=0)
        torch.save(weighted_sentence, 'weighted_sentence.pth')

    def load_glove_model(self, glove_file):
        f = open(glove_file, 'r', encoding="utf8")
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]], dtype=np.float16)
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

    def cosine_distance(self, vector1, vector2, word2vec):
        vector1 = word2vec[vector1]
        vector2 = word2vec[vector2]
        inner_product = np.sum(np.dot(vector1, vector2))
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cosine_distance = 1 - inner_product / magnitude
        return cosine_distance
