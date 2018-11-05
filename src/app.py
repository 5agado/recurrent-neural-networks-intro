from model.textGenerator import TextGenerator
import configparser
import sys
import os


from flask import Flask, request
app = Flask(__name__)

config = configparser.ConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.ini'))


@app.route('/models/<string:model_id>/generate', methods=['GET'])
def generate_text(model_id):
    args = request.args
    text_gen = TextGenerator(config.get('base', 'MODELS_CONFIG'),
                            model_id,
                            args.get('model_config', 'standard_config'))

    return text_gen.generate(int(args.get('min_nb_words', 10)),
                             seed=args.get('seed', None))


if __name__ == '__main__':
    app.run()
