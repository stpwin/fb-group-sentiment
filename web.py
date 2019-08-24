from flask import Flask, render_template, abort, request, jsonify

app = Flask(__name__)

@app.route('/api/word_tokenizer', methods=["GET"])
def word_tokenizer_api():
    sent = request.args.get('sent', 0, type=str)
    tokenized = ["Hello", "What"]
    return jsonify(result='|'.join(tokenized))

@app.route('/api/classify', methods=["GET"])
def classify_api():
    input_str = request.args.get('input_str', 0, type=str)
    simple = ["contains(Hell)", "contains(So)"]
    tokenized = ["Hello", input_str]
    entities = [_str[8:-1] for _str in simple]
    prop = {
        "pos": 0.0000,
        "neg": 0.0000,
        "neu": 0.0000
    }
    return jsonify(tokenized='|'.join(tokenized), entities="|".join(entities), prop=prop)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80, debug=True)
