import os
import sys

import configparser

import logging
from utils import model_utils
from model.textGenModel import TextGenModel


class TextGenerator:

    def __init__(self, config_filepath, model_id, config_id):
        """
        Initiate a proxy for a text generation model.
        :param model_id: section in configuration file with info about model data
        :param config_id: section in configuration file with info about model parameters
        """

        # load models configuration file
        config = configparser.ConfigParser()
        config.read(config_filepath)
        # exit if there is no info for the given model ID
        if model_id not in config:
            logging.error("{} no such model".format(model_id))
            exit(1)
        # load and configure model
        else:
            # TODO add option of local VS remote loading
            if config.has_option(model_id, 'host'):
                self.text_gen_model = self._load_proxy(config, model_id, config_id)
            else:
                self.text_gen_model = self._load_model(config, model_id, config_id)

    def _load_proxy(self, config, model_id, config_id):
        # additional configuration settings to be fed to the model
        # settings = {el[0]:el[1] for el in config.items(config_id)}
        settings = {'sent_max_len': config.getint(config_id, 'sent_max_len'),
                    'temperature': config.getfloat(config_id, 'temperature'),
                    'use_embeddings': config.getboolean(config_id, 'use_embeddings')}
        return model_utils.load_model_proxy(
            config.get(model_id, 'config_yaml_path'),
            config.get(model_id, 'index_to_word_path'),
            config.get(model_id, 'host'),
            config.get(model_id, 'port'),
            model_type=config.get(model_id, 'type'),
            **settings)

    def _load_model(self, config, model_id, config_id):
        # additional configuration settings to be fed to the model
        # settings = {el[0]:el[1] for el in config.items(config_id)}
        settings = {'sent_max_len': config.getint(config_id, 'sent_max_len'),
                    'temperature': config.getfloat(config_id, 'temperature'),
                    'use_embeddings': config.getbool(config_id, 'use_embeddings')}
        return model_utils.load_model_local(
            config.get(model_id, 'model_path'),
            config.get(model_id, 'weights_path'),
            config.get(model_id, 'index_to_word_path'),
            model_type=config.get(model_id, 'type'),
            **settings)

    def generate(self, text_min_len, text_max_len=None, seed=None):
        """
        Get generated text. Text is generated my the model and then prettified to string.
        :param text_min_len: minimum length for a valid generated text
        :param text_max_len: maximum text length (generated text will be cut)
        :param seed: seed text to use for the generation task
        """
        # get raw generated text
        gen_text = self.text_gen_model.get_sentence(text_min_len, seed)
        # pretty print text and return
        pretty_text = self.text_gen_model.pretty_print_sentence(gen_text, text_max_len)
        return pretty_text
